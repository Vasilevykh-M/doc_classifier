from pathlib import Path
import tensorrt as trt
import torch
from model.utils import get_processor
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from PIL import Image

class TRTModel:
    def __init__(self, model_path):
        model_math = Path(model_path)
        self.processor = get_processor()
        if model_math.suffix != ".engine" or not model_math.exists():
           model_path = self.build(model_path)

        self.engine = self._load_engine(model_path)
        self.context = self._create_context()
        self.buffers = {"host": {}, "device": {}}
        self.stream = cuda.Stream()
        self._setup_buffers()
        self.io_info = self.get_io_info()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build(self, model_path):
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        with open(model_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("Engine build failed")
        
        with open(f"{model_path[:-5]}.engine", "wb") as f:
            f.write(engine)

        return f"{model_path[:-5]}.engine"
    
    def _load_engine(self, model_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(engine_data)
            

    def _create_context(self):
        return self.engine.create_execution_context()
    
    def _setup_buffers(self):
        """Настройка буферов ввода/вывода"""
        self.buffers = {"host": {}, "device": {}}

        tensor_names = []
        for i in range(self.engine.num_io_tensors):
            tensor_names.append(self.engine.get_tensor_name(i))
        
        for name in tensor_names:
            tensor_type = self.engine.get_tensor_mode(name)
            
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if -1 in shape:
                min_shape = self.engine.get_tensor_profile_shape(name, self.profile_idx, trt.ProfileFormat.OPT)
                shape = min_shape[1]
            
            np_dtype = trt.nptype(dtype)
            host_buffer = np.empty(shape, dtype=np_dtype)
            
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            
            self.buffers["host"][name] = host_buffer
            self.buffers["device"][name] = device_buffer
            
            self.context.set_tensor_address(name, int(device_buffer))
            
            print(f"Buffer allocated: {name}, shape: {shape}, dtype: {np_dtype}, size: {host_buffer.nbytes} bytes")

    def set_input_shape(self, tensor_name, shape):
        if tensor_name not in self.buffers["host"]:
            raise ValueError(f"Tensor {tensor_name} not found")
            
        self.context.set_input_shape(tensor_name, shape)
        
        current_shape = self.buffers["host"][tensor_name].shape
        if current_shape != shape:
            dtype = self.buffers["host"][tensor_name].dtype
            
            if self.buffers["device"][tensor_name]:
                self.buffers["device"][tensor_name].free()
                
            host_buffer = np.empty(shape, dtype=dtype)
            device_buffer = cuda.mem_alloc(host_buffer.nbytes)
            
            self.buffers["host"][tensor_name] = host_buffer
            self.buffers["device"][tensor_name] = device_buffer
            
            self.context.set_tensor_address(tensor_name, int(device_buffer))

    def get_io_info(self):
        """Получение информации о входах/выходах"""
        io_info = {"inputs": [], "outputs": []}
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            
            info = {
                "name": name,
                "dtype": dtype,
                "shape": shape
            }
            
            if mode == trt.TensorIOMode.INPUT:
                io_info["inputs"].append(info)
            elif mode == trt.TensorIOMode.OUTPUT:
                io_info["outputs"].append(info)
            
        return io_info

    def preprocess(self, img_path):
        x = Image.open(img_path)
        x = self.processor(x)

        input_dtype = self.io_info['inputs'][0]['dtype']
        input_name = self.io_info['inputs'][0]['name']

        input_data = {
            input_name: x[None, :].numpy().astype(input_dtype)
        }

        return input_data


    def infer(self, input_data):
        start_time = time.time()
        
        # 1. Подготовка входных данных
        for name, data in input_data.items():
            if name not in self.buffers["host"]:
                raise ValueError(f"Input tensor {name} not found")
                
            # Проверка формы
            current_shape = self.buffers["host"][name].shape
            if data.shape != current_shape:
                self.set_input_shape(name, data.shape)
                
            # Копирование данных
            np.copyto(self.buffers["host"][name], data)
            
            # Копирование на устройство
            cuda.memcpy_htod_async(
                self.buffers["device"][name],
                self.buffers["host"][name],
                self.stream
            )
        
        # 2. Выполнение инференса
        self.context.execute_async_v3(self.stream.handle)
        
        # 3. Копирование результатов с устройства
        for name, buffer in self.buffers["host"].items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                cuda.memcpy_dtoh_async(
                    buffer,
                    self.buffers["device"][name],
                    self.stream
                )
        
        # 4. Синхронизация
        self.stream.synchronize()
        inference_time = time.time() - start_time
        
        # 5. Формирование результатов
        results = {}
        for name, buffer in self.buffers["host"].items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                results[name] = buffer.copy()
        print(f"Inf time: {inference_time}")
        return torch.from_numpy(results[self.io_info["outputs"][0]["name"]]).to(self.device)

    def __call__(self, x):
        x = self.preprocess(x)
        result = self.infer(x)
        return result
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, "buffers") and "device" in self.buffers:
            for buffer in self.buffers["device"].values():
                if buffer:
                    buffer.free()
        if hasattr(self, "context") and self.context:
            del self.context
        if hasattr(self, "engine") and self.engine:
            del self.engine
        print("Resources released")

class HostDeviceMem:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.host = cuda.pagelocked_empty(size, dtype)
        self.device = cuda.mem_alloc(self.host.nbytes)

    def free(self):
        if self.device:
            self.device.free()
            self.device = None
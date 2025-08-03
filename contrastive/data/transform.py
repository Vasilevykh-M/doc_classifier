import albumentations as A
import cv2

def get_train_transform(target_size):
    return A.Compose([
        A.LongestMaxSize(max_size=target_size.max_shape(), p=1),

        A.PadIfNeeded(
            min_height=target_size.height,
            min_width=target_size.width,
            border_mode=cv2.BORDER_CONSTANT,
            p=1
        ),

        A.OneOf([
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
        ], p=0.7),

        # Яркость/контраст
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            brightness_by_max=True,
            p=0.5
        ),

        # Цветовые искажения
        A.OneOf([
            A.ChannelShuffle(p=0.3),
            A.ToGray(p=0.1),
            A.CLAHE(clip_limit=3.0, p=0.2),
            A.FancyPCA(alpha=0.2, p=0.2),
        ], p=0.5),

        # Цветовые шумы
        A.OneOf([
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=(0.1, 0.5),
                p=0.3
            ),
            A.GaussNoise(p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),
        ], p=0.4),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.MedianBlur(blur_limit=5, p=0.3),
        ], p=0.5),

        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        A.OpticalDistortion(
            distort_limit=0.1,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),

        # Аугментации качества изображения
        A.CoarseDropout(
            p=0.3
        ),
        A.RandomSunFlare(
            src_radius=100,
            src_color=(255, 255, 255),
            p=0.1
        ),
    ])

def get_val_transform(target_size):
    return A.Compose([
        A.LongestMaxSize(max_size=target_size.max_shape(), p=1),
        A.PadIfNeeded(
            min_height=target_size.height,
            min_width=target_size.width,
            border_mode=cv2.BORDER_CONSTANT,
            p=1
        ),
    ])
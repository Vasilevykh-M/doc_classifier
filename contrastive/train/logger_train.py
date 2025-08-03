from matplotlib import pyplot as plt


class Logger:
    def __init__(self):
        self.epoch = []
        self.val_metrics = {}
        self.train_metrics = {}

    def add_metrics(self, name):
        self.val_metrics[name] = []
        self.train_metrics[name] = []

    def __call__(self, epoch, name, value_val, value_tr):
        if not epoch in self.epoch:
            self.epoch.append(epoch)
        self.val_metrics[name].append(value_val)
        self.train_metrics[name].append(value_tr)

    def draw(self, name):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.epoch, self.train_metrics[name], 'b-', label=f'Train {name}')
        ax.plot(self.epoch, self.val_metrics[name], 'r-', label=f'Validation {name}')
        
        # Настраиваем оформление
        ax.set_title(f'Training and Validation {name}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(name)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Автоматическая настройка осей
        ax.relim()
        ax.autoscale_view()
        
        plt.tight_layout()
        plt.show()

        return fig

    def save(self, name, path):
        fig = self.draw(name)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
from tensorboardX import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, log_dir):
        '''
        Extended SummaryWriter Class from tensorboard-pytorch (tensorboardX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer
        '''

        super().__init__(log_dir)
        self.log_dir = self.file_writer.get_logdir()
    
    def update_parameters(self, module, step_i):
        '''
        module: nn.Module
        '''

        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        self.add_histogram(name, values, step_i)
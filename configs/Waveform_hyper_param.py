class Param(object):
    def __init__(self):
        # model configs
        self.batch_size = 256
        self.lr = 0.001
        self.repr_dims = 512
        self.epochs = 40
        self.num_cluster = '4,8'
        self.backbone_type = 'TS_CoT'
        self.model_path = "pretrained_model/Waveform_model.pkl"
 


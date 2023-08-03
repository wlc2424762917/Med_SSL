from models.Med_simMIM import SwinTransformerForSimMIM
from models.Med_simMIM import SimMIM


def build_MIM_encoder(config, type='swin'):
    if type == 'swin':
        # encoder = SwinTransformerForSimMIM(
        #     img_size=config.DATA.IMG_SIZE,
        #     patch_size=config.MODEL.SWIN.PATCH_SIZE,
        #     in_chans=config.MODEL.SWIN.IN_CHANS,
        #     num_classes=0,
        #     embed_dim=config.MODEL.SWIN.EMBED_DIM,
        #     depths=config.MODEL.SWIN.DEPTHS,
        #     num_heads=config.MODEL.SWIN.NUM_HEADS,
        #     window_size=config.MODEL.SWIN.WINDOW_SIZE,
        #     mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        #     qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        #     qk_scale=config.MODEL.SWIN.QK_SCALE,
        #     drop_rate=config.MODEL.DROP_RATE,
        #     drop_path_rate=config.MODEL.DROP_PATH_RATE,
        #     ape=config.MODEL.SWIN.APE,
        #     patch_norm=config.MODEL.SWIN.PATCH_NORM,
        #     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        encoder = SwinTransformerForSimMIM(
            img_size=224,
            patch_size=4,
            in_chans=1,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            drop_path_rate=0.2,
            ape=False,
            patch_norm=True,
            use_checkpoint=False)
        encoder_stride = 32
        model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)
    else:
        raise NotImplementedError
    return model
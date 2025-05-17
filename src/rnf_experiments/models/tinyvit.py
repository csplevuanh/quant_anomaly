import timm

def tinyvit_iot_s4(pretrained: bool = True):
    """Load TinyViT‑IoT‑S4 vision model."""
    return timm.create_model('tiny_vit_5m_224', pretrained=pretrained)

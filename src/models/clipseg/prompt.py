import torch
from src.third_party.openai_clip import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

ln_prompt = torch.cat(
    [
        clip.tokenize(
            """Benign lymph node: Oval shape, preserved echogenic hilum, thin homogeneous cortex. Malignant lymph node: Round, lost hilum, thickened/heterogeneous cortex, microcalcifications, irregular margins."""
        )
    ]
).to(device)

busi_prompt = torch.cat(
    [
        clip.tokenize(
            """Benign breast lesion: Oval shape, smooth margins, parallel orientation, homogeneous hypoechoic echotexture, posterior enhancement. Malignant breast lesion: Irregular shape, spiculated margins, non-parallel orientation, heterogeneous hypoechoic echotexture, microcalcifications, posterior shadowing."""
        )
    ]
).to(device)

thyroid_prompt = torch.cat(
    [
        clip.tokenize(
            """Benign thyroid nodule: oval, wider-than-tall, homogeneous, smooth margins, intact capsule. Malignant thyroid nodule: taller-than-wide, hypoechoic, irregular margins, microcalcifications, capsular/extra-thyroidal invasion."""
        )
    ]
).to(device)

prostate_prompt = torch.cat(
    [
        clip.tokenize(
            """Benign prostate: smooth, symmetric TZ enlargement with heterogeneous nodules and intact capsule; Malignant prostate: focal peripheral-zone hypoechoic lesion with irregular margins, capsular breach and increased Doppler flow."""
        )
    ]
).to(device)

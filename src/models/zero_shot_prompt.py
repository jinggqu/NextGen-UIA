# Define ensemble prompts optimized for medical context
LN_PROMPTS_ENSEMBLE = {
    "benign": [
        "A benign lymph node with an oval shape and a preserved fatty hilum",
        "A benign lymph node with a long-to-short axis ratio greater than 2",
        "A benign lymph node showing a clear, echogenic central hilum",
        "A benign lymph node with a smooth, well-defined border",
        "A benign lymph node characterized by its regular, oval morphology and homogeneous echotexture",
        "A benign lymph node with a thin, uniform cortex surrounding a prominent hilum",
        "A benign lymph node appearing as a well-defined, hypoechoic oval structure with a bright central hilum",
        "A benign lymph node featuring a distinct fatty hilum and regular shape",
        "A benign lymph node with normal morphology, including a visible hilum and uniform cortex",
        "A benign lymph node that is distinctly elongated and maintains its central echogenic hilum",
    ],
    "malignant": [
        "A malignant lymph node with a round shape and an absent or effaced hilum",
        "A malignant lymph node with a long-to-short axis ratio less than 2",
        "A malignant lymph node with loss of the central fatty hilum",
        "A malignant lymph node with an irregular, spiculated, or blurred border",
        "A malignant lymph node containing internal microcalcifications",
        "A malignant lymph node showing internal cystic necrosis or liquefaction",
        "A malignant lymph node that is markedly hypoechoic and has a heterogeneous texture",
        "A malignant lymph node with eccentric cortical thickening",
        "A malignant lymph node appearing as a round, solid mass with indistinct margins",
        "A malignant lymph node characterized by a round shape and heterogeneous internal echoes",
    ],
}

BREAST_PROMPTS_ENSEMBLE = {
    "benign": [
        "A benign nodule with an oval shape and circumscribed margins",
        "A benign nodule with a parallel orientation, appearing wider-than-tall",
        "A benign nodule, simple cyst which is anechoic with posterior acoustic enhancement",
        "A benign nodule that is well-circumscribed and has a homogeneous echo pattern",
        "A benign nodule with a smooth border and an oval shape",
        "A benign nodule appearing as a solid, oval, and circumscribed mass",
        "A benign nodule with a gently lobulated but well-defined margin",
        "A benign nodule that is isoechoic and has a distinct, thin echogenic capsule",
        "A benign nodule with an oval shape, parallel orientation, and circumscribed margin",
        "A benign nodule with regular morphology and well-defined borders",
    ],
    "malignant": [
        "A malignant nodule with an irregular shape and spiculated margins",
        "A malignant nodule with a non-parallel orientation, appearing taller-than-wide",
        "A malignant nodule causing posterior acoustic shadowing",
        "A malignant nodule with indistinct or angular margins",
        "A malignant nodule containing internal microcalcifications",
        "A malignant nodule that is markedly hypoechoic and has an irregular shape",
        "A malignant nodule with a heterogeneous echo pattern and ill-defined borders",
        "A malignant nodule with microlobulated margins",
        "A malignant nodule that is irregular in shape and demonstrates posterior shadowing",
        "A malignant nodule with suspicious morphology, including an irregular shape and non-circumscribed margins",
    ],
}

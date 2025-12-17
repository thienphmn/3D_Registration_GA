The dataset used for this 3D rigid registration is taken from the Retrospective Image Registration Evaluation Project (RIRE).
It can be found under the following link: https://rire.insight-journal.org/

Preprocessing of CT volumes:
- intensity normalization
- intensity windowing (Window Level = 40, Window Width = 400) -> equal to soft tissue window
- gaussian blurring

Preprocessing of MRI volumes:
- intensity normalization
- gaussian blurring
- padding in all dimension with 25 voxels with constant value of 0.0

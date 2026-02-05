<div align="center">

# A Method for Re-Generating Vehicle Speed Trajectories to Connect Traffic Microsimulation with Vehicle Emission Models


[Shaomin Qin](https://github.com/qinshaomin77)· [Haobing Liu](https://scholar.google.com/citations?user=e-8R2vMAAAAJ&hl=en)


</div>

## Prerequisites

- Windows 10/11（x64）
- Python >=3.9
- SUMO 
- Gurobi

## Dependencies

This project requires the following external tools in addition to Python packages.

### 1) SUMO (Eclipse SUMO)
Please install SUMO and make sure `sumo` / `sumo-gui` are available in your system PATH.
- Official installation guide: https://sumo.dlr.de/docs/Downloads.php
- Documentation: https://sumo.dlr.de/docs/
  
### 2) Gurobi Optimizer (License required)
Please install SUMO and make sure `Gurobi`, and activate a license (academic or commercial).
- Download & installation: https://www.gurobi.com/downloads/
- License request / activation: https://www.gurobi.com/downloads/end-user-license-agreement/
- Academic license info: https://www.gurobi.com/academia/academic-program-and-licenses/
Install python package dependencies through pip:

```bash
$ pip install -r requirements.txt
```


# 💻 Run Demo


In the `Cad_VLM/config/inference_user_input.yaml`, provide the following path.

<details><summary>Required Updates in yaml</summary>
<p>

- `cache_dir`: The directory to load model weights from Huggingface.
- `log_dir`: Directory for saving _logs, outputs, checkpoints_.
- `checkpoint_path`: The path to model weights.

</p>
</details> 
<br>

```bash
$ cd App
$ gradio app.py
```

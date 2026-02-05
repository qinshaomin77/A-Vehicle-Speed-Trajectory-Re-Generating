<div align="center">

# A Method for Re-Generating Vehicle Speed Trajectories to Connect Traffic Microsimulation with Vehicle Emission Models


[Shaomin Qin](https://github.com/qinshaomin77)· [Haobing Liu](https://scholar.google.com/citations?user=e-8R2vMAAAAJ&hl=en)


</div>


# ⚙️ Installation


## Prerequisites

- Windows 10/11（x64）
- Python >=3.9
- SUMO 
- Gurobi

## 📦 Dependencies

```bash
$ conda create -n rvst python=3.9
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

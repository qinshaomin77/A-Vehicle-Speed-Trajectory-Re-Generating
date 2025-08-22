<div align="center">

# A Vehicle Speed Trajectory Re-Generating Toolkit Connecting Traffic Microsimulation Tools with Vehicle Emission Models


[Qin Shaomin](https://github.com/qinshaomin77)· [Liu Haobing](https://scholar.google.com/citations?user=e-8R2vMAAAAJ&hl=en)


</div>


# ⚙️ Installation

## 🌍 Environment

- 🪟 Windows 10/11（x64）
- 🐍 Python >=3.9

## 📦 Dependencies

```bash
$ conda create -n retraj python=3.8
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

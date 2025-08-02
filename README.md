# PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification
*Magdalena Trędowicz, Łukasz Struski, Marcin Mazur, Szymon Janusz, Arkadiusz Lewicki, Jacek Tabor*

[PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification](https://arxiv.org/abs/2406.11443)


---
Abstract: Video processing is generally divided into two main categories: processing of the entire video, which typically yields optimal classification outcomes, and real-time processing, where the objective is to make a decision as promptly as possible. Although the models dedicated to the processing of entire videos are typically well-defined and clearly presented in the literature, this is not the case for online processing, where a plethora of hand-devised methods exist. To address this issue, we present PrAViC, a novel, unified, and theoretically-based adaptation framework for tackling the online classification problem in video data. The initial phase of our study is to establish a mathematical background for the classification of sequential data, with the potential to make a decision at an early stage. This allows us to construct a natural function that encourages the model to return a result much faster. The subsequent phase is to present a straightforward and readily implementable method for adapting offline models to the online setting using recurrent operations. Finally, PrAViC is evaluated by comparing it with existing state-of-the- art offline and online models and datasets. This enables the network to significantly reduce the time required to reach classification decisions
while maintaining, or even enhancing, accuracy.

---
<img src="images/Teaser_PrAViC.pdf" width="600">


### Installation
The following installation instructions are provided for a Conda-based Python environment.

```shell
git clone https://github.com/mtredowicz/PrAViC.git
cd PrAViC
pip install -r requirements.txt
```
### Repository Structure

```bash
PrAViC/
├── network/               # Online models: R3D and S3D
├── configs/               # [Hydra](https://hydra.cc) YAML configs to define all training and evaluation settings
├── dataloaders/           # Dataset loading utilities
├── images/                # Images and visualizations
├── losses/                # Custom cross entropy and fast cross entropy implementation
├── main.py                # Main script file for training
└── README.md              # Project documentation
```
### Running a Demo

We provide example configs for training online versions of R3D and S3D models on UCF101 dataset.

Make sure you run the commands from the project root directory.

To train using the default configuration:
```bash
# Train with base values included in the configs/demo.yaml
python main.py --config-name demo
```
### Citations

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
<h3 class="title">PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification</h3>
    <pre><code>@misc{trkedowicz2024pravic,
      title={PrAViC: Probabilistic Adaptation Framework for Real-Time Video Classification},
      author={Tr{\k{e}}dowicz, Magdalena and Struski, {\L}ukasz and Mazur, Marcin and Janusz, Szymon and Lewicki, Arkadiusz and Tabor, Jacek},
      year={2024},
      eprint={2406.11443},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.11443},
}

</code></pre>
</section>

# RoleFact

Code for EMNLP 2024 findings paper: [Mitigating Hallucination in Fictional Character Role-Play](https://arxiv.org/abs/2406.17260).  In this work, we focus on the evaluation and mitigation of hallucination in fictional character role-play. We introduce the Script Grounded Roleplay (SGR) dataset with more than 2,000 characters and 72,000 interviews, including 18,000 adversarial questions. We propose RoleFact, a role-playing method that mitigates hallucination by modulating the influence of parametric knowledge using a pre-calibrated confidence threshold.

## Code

Coming soon

## Dataset

Download the dataset from https://drive.google.com/drive/folders/14sC2gN4hWjs2TmL_kpmsVtJZ1hUhXYDk?usp=sharing

Extract the dataset within rolefact directory.

```
rolefact
├── sgr/
│   ├── script_knowledge/
│   ├── tasks/
│   │   ├── adversarial_interview
│   │   └── open_ended_interview
│       ├── dialogue_completion
│       └── scene_grounded_interview
```

## Explore data

To find all story specific files, you can run the following code.

```
import os
story_files = os.listdir("./sgr/script_knowledge")
print(story_files[:5]) 
```

To load script specific knowledge for a specific story

```
from script_kb import ScriptKB
story_kb = ScriptKB("./sgr/script_knowledge/M0507.json")
print(story_kb.get_title())
print(story_kb.get_characters()[:5])
print(story_kb.get_character_profile("HICCUP")['second_person'])
print(story_kb.get_kb_between('M000U0068','M000U0071'))
```

To explore task examples such as adversarial_interview

```
import json
with open("./sgr/tasks/adversarial_interview/M0507.json",'r') as infile:
    adv_content = json.load(infile)
print(len(adv_content))
print(adv_content[1])
```

See the data_explore notebook for more details.

## Citation
If you use the dataset, please cite the following work.

```bibtex
@misc{sadeq2024mitigatinghallucinationfictionalcharacter,
      title={Mitigating Hallucination in Fictional Character Role-Play}, 
      author={Nafis Sadeq and Zhouhang Xie and Byungkyu Kang and Prarit Lamba and Xiang Gao and Julian McAuley},
      year={2024},
      eprint={2406.17260},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.17260}, 
}
```




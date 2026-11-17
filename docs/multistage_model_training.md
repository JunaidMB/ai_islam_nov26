# Multistage Model Training

* Modern Language Models are trained in multiple stages. There is typically pre-training and post-training but more recently, a mid-training stage has been introduced that sits between them.
* For a model like GLM 4.5 the multistage training pipeline is

![[pretraining_and_midtraining.png]]

The output of the pre and mid training stages is a called a base model. The post-training stage will take the base model and perform finetuning in stages. For GLM, the post-training stage is

![[post-training.png]]
# SenceClassification-AIChallenge
Last modified by Xiaodong Wu 2017.12.04
Contact me by xiaodong[dot]wu[dot]c[at]gmail.com

AI Challenge 2017
Scene Classificationa task

testb_results_1203.json  # the final result we submitted.
datasets   
finetuned-models # you can run the ./show_results/show_results.py to check the resutls

Except the models in ./finetuned-models I have also conduct the following trail:
1. focal loss
    See it in github: https://github.com/xiaodongww/pytorch/blob/master/focal_loss.py
    Know more about focal loss:  https://arxiv.org/abs/1708.02002
2. Truncated loss 
    Self defined loss, the same idea as focal loss. Trying to decay the loss of well calssified samples,
    and try to let the model focus on the hard smaples. In this loss function, I just set the loss of 
    samples whose max probability after softmax layer is bigger than 0.5 to be 0. The result is not as good as I thought.
    See it in github: https://github.com/xiaodongww/pytorch/blob/master/focal_loss.py
3. Combine the SDD detection resutls.
    In this trail, I concated the probability result of SSD detection and the original CNN features(features of 161).
    The result is not very good. 
    During this experiment, I also write a script to load pictures.  https://github.com/xiaodongww/pytorch/blob/master/diy_dataset.py
4. Cascade prediction
    The idea is coming from the insight that maybe the false classified samples have an unconfident probability distribution (i.e. the output of softmax layer). I trained a model specially for all the low confident pictures(max prob<0.5). The model is the same as the main model. The loss of this model did not decay during training. So I give up this idea. (Maybe the model should have a different architecture from the main model.)
5. In the experiment of last few days before the deadline, I test the following tricks.
    1) Data augmentation  https://github.com/twtygqyy/caffe-augmentation
    2) 10-crop test   https://github.com/xiaodongww/caffe/blob/master/multi_crop_test.py
    3) Model ensembling

Something about the learning rate:
For the finetuning process, I find that the result is very sensitive to the original learning rate.
The initial learning rate can be 1e-5~1e-7. 1e-3 and 1e-4 did not achieve the best result in my experiment.
For the learing rate decay policy, I do not recommend the 'step' policy. I think it is better to set a relative small initial value, and keep the value until the loss does not decay or the test result begin to decay. Then deacay the learning rate.

Something about overfitting:
Some models face problems of overfitting. The trainging loss is always decaying and the accuracy promotes first and then deacy. In my experiment, I just check the log file, and find the best model in history. And then I will decay the learning rate and continue training based on the finded the best model.

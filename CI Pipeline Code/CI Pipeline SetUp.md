# CI Pipeline Set up 

CI pipeline will also be included in the repo just to keep in one place since I am the only one working on it. 

The goal of this is to give users a step by step guide on how to set up the git runner for this specific pipeline before I forget how to do it years later 

#### Preq 

1. Nvidia Jetson Orin Nano already set up and working. 
    - Follow the Jetson guide for set up 

2. Git ssh key already done 
    - Follow this guide to get it done https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
3. Download hugging face docker image from private repo. 
    - This will already have pytorch and packages working along with important set up features 
    - *Will need to update as I go through it 
    - Will have to login to hugging face to get the model downloaded 
    hf auth login 
    
    hf download BlueDoorE/edith-glasses_jp64.tar edith-glasses_jp64.tar
    Have hf auth and can download using the command above 

4. Docker Installed 
    - Will need to load the docker image from the tar itself 
5. Github runner 
    - This will be used to run for each pull request 
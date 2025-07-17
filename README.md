###  **Stochastic Interpolant Based Flow & Diffusion**


This project implements stochastic interpolants based diffusion and flow models. 
A lot of it is inspired by the following papers

- [An introduction to Flow and Diffusion Models](https://arxiv.org/abs/2506.02070)
- [Stochastic Interpolants, a unifying framework for flow and diffusion models](https://arxiv.org/abs/2303.08797)

***
 
The model is trained using flow matching, which directly learns the vector field $u_t^{\theta}(x)$, not the noise or score function. 

To sample from the flow model, an ODE solver is used on the vector field(Euler)

To sample from the diffusion model, an SDE solver is used. 

Classifier-free guidance is also implemented 
***

### Usage
To recreate the project, follow these steps:
Create a folder named `diffusion_project` and clone this repository inside it
```bash 
git clone git@github.com:Blaise143/stochastic-interpolant-diffusion.git
cd diffusion_project 
```

#### Training the Vector Field

To train the vector field for the flow and diffusion model without guidance:
```bash
python main.py --mode train_flow --epochs 100 --batch_size 200 --lr 1e-3
```

To train with classifier-free guidance (for conditional generation):
```bash
python main.py --mode train_flow --epochs 100 --batch_size 200 --lr 1e-3 --guided --guidance_scale 2.0
```
***
### Sampling from the Trained Vector Field

#### Flow-based Sampling (Euler ODE Solver)

To generate samples using the flow model with Euler ODE solver:
```bash
python main.py --mode sample --model_type flow --num_samples 10 --num_steps 500 --guided --guidance_scale 2.0 --save_path flow_samples.png
```

You can specify particular digits to generate:
```bash
python main.py --mode sample --model_type flow --digits "0,1,2" --num_samples 5 --guided --guidance_scale 2.0
```

#### Diffusion-based Sampling (SDE Solver)

To generate samples using the diffusion model with Euler-Maruyama SDE solver:
```bash
python main.py --mode sample --model_type diffusion --num_samples 10 --num_steps 500 --guided --guidance_scale 2.0 --save_path diffusion_samples.png
```

For unguided diffusion sampling:
```bash
python main.py --mode sample --model_type diffusion --num_samples 10 --num_steps 500
```
***

Alright, see ya later mate 


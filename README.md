**Project Structure**

This project implements a stochastic interpolant based diffusion and flow model. 

Classifier free guidance and classifier guidance is well integrated with just the flow model so far


***
**NOTES**
- The model is trained using flow matching
- The U-NET directly predicts the vector field $u_t^{\theta}$, not the noise $\epsilon_t^{\theta}$
***

Since we are using a gaussian probability path, you can easily convert the vector field to the scores and sample from it using `euler maruyama` method or any other SDE solver.

You can use the following conversion:
$$
s_t^{\theta}(x) = \frac{\alpha_t u_t^{\theta}(x)-\dot{\alpha_t}x}{\beta_t^2 \dot{\alpha_t}-\alpha_t \dot{\beta_t} \beta_t}
$$
Note that we are using the CondOT path where $\alpha_t= t$ and $\beta_t = 1-t$

So the simplified conversion is:
$$
s_t^{\theta}(x) = \frac{t  u_t^{\theta}(x)-x}{(1-t)^2+ t (1-t)}
$$
In this case, the corresponding Stochastic Differential Equation is: 
[TODO]

In order to recreate the project, follow the following steps..
```bash 
git clone <current_repository>
cd diffusion_project 
```
## Usage

### Training the Flow Model

To train the flow model without guidance run the following command
```bash
python main.py --mode train_flow --epochs 100 --batch_size 200 --lr 1e-3
```

To train the flow model with classifier-free guidance run the following command:

You can adjust the batch size as depending on whether or not your machine got it.

```bash
python main.py --mode train_flow --epochs 100 --batch_size 200 --lr 1e-3 --guided --guidance_scale 2.0
```

### Training The Noisy Classifier Model

To train a classifier model for guided sampling run the following command:

```bash
python main.py --mode train_classifier --epochs 100 --batch_size 200 --lr 1e-3
```

### Classifier-guidance samples 

To generate samples using classifier guidance run the following command:
You can also adjust the hyperparameters to your liking.
```bash
python main.py --mode sample --num_samples 10 --num_steps 500 --guidance_scale 2.0 --save_path samples.png
```
***



$$u_t^θ(x) = (β_t^2  \frac{α̇_t}{α_t} -  \dot{β_t} β_t)  s_t^θ(x) + \frac{α̇_t}{α_t}  x$$
$$u_t^θ(x_t)=\frac{x_t-\epsilon_t^\theta(x_t)}{t}$$ 




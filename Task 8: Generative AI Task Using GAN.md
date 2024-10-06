# Task 8: GAN Implementation Using CIFAR-10 Dataset

### 1. **Introduction**
In this task, we implemented a Generative Adversarial Network (GAN) to generate synthetic images using the CIFAR-10 dataset. CIFAR-10 is a widely used dataset consisting of 60,000 32x32 color images across 10 different classes. The aim was to train a GAN model to produce realistic images that mimic the CIFAR-10 classes. The GAN architecture consists of two primary models:
- **Generator**: This model generates synthetic images from random latent vectors.
- **Discriminator**: This model evaluates whether the generated images are real (from the dataset) or fake (from the generator).

### 2. **Code Implementation**

Here is the Python code used to develop and train the GAN model using Keras and TensorFlow:

```python
from numpy import ones, zeros, randn
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(trainX, _), (_, _) = cifar10.load_data()

# Plot 25 sample images from the dataset
for i in range(25):
    plt.subplot(5, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainX[i])
plt.show()

# Define the discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Define the generator model
def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 8 * 8
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, (8, 8), activation='tanh', padding='same'))
    return model

# Define the GAN combining generator and discriminator
def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# Load and preprocess CIFAR-10 dataset
def load_real_samples():
    (trainX, _), (_, _) = cifar10.load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5
    return X

# Generate real samples from the dataset
def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y

# Generate latent points as input for the generator
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# Generate fake images using the generator
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y

# Train the GAN model
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss_real, _ = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss_fake, _ = d_model.train_on_batch(X_fake, y_fake)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'Epoch {i+1}, Batch {j+1}/{bat_per_epo}, d_loss_real={d_loss_real:.3f}, d_loss_fake={d_loss_fake:.3f}, g_loss={g_loss:.3f}')
    g_model.save('cifar_generator_model.h5')

# Define model parameters
latent_dim = 100
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
dataset = load_real_samples()

# Train the GAN model
train(generator, discriminator, gan_model, dataset, latent_dim, n_epochs=2)

# Load the trained generator model and generate images
from keras.models import load_model

def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, :])
    plt.show()

model = load_model('cifar_generator_model.h5')
latent_points = generate_latent_points(100, 25)
X = model.predict(latent_points)
X = (X + 1) / 2.0

show_plot(X, 5)
```

### 3. **Training Process**
The GAN model was trained for **2 epochs** on the CIFAR-10 dataset. During the training process, the discriminator and generator losses were recorded for each batch. The discriminator loss measures its ability to correctly classify real and fake images, while the generator loss indicates how well the generator is performing at fooling the discriminator.

The model was trained with the following settings:
- **Latent vector dimension**: 100
- **Batch size**: 128
- **Epochs**: 2

The training loop updates the discriminator on real and fake images separately for better performance. The generator tries to produce images that can trick the discriminator, and the GAN loss indicates how well it performs.

### 4. **Generated Images**
The output of the trained generator after 2 epochs is shown below. The model generated 25 images representing various categories in the CIFAR-10 dataset. These include objects like trucks, birds, horses, ships, etc.

![image](https://github.com/user-attachments/assets/9fefbb2e-2e05-4b27-b855-71ed9f3df349)


### 5. **Conclusion**
This project successfully demonstrates the use of a simple GAN model for generating images from the CIFAR-10 dataset. The model was able to generate diverse images resembling different classes in CIFAR-10, even with only 2 epochs of training. However, with additional training and fine-tuning of the GAN architecture, further improvements in image quality could be achieved.

### 6. **Future Work**
- **Training for more epochs**: Increasing the number of epochs would allow the model to generate more refined and higher-quality images.
- **Conditional GAN**: Implementing a conditional GAN (CGAN) would allow control over the image class generated by the model.
- **Advanced architectures**: Exploring more advanced GAN architectures, such as DCGAN or Wasserstein GAN, could further enhance the image generation quality.

This implementation serves as a solid foundation for generating synthetic images and can be expanded further for other generative tasks.

---



COVID-19 testing is inadequate, especially in developing countries. Testing is scarce, requires trained nurses with costly equipment, and is expensive, limiting how many people can obtain their results. Also, many people in developing countries cannot risk taking tests because results are not anonymous, and a positive result may mean a loss of work income and starvation, which further allows COVID-19 to spread. 

Numerous attempts have been made to solve this problem with partial success, including contact tracing apps which have not been widely adopted often due to privacy concerns. Pharmaceutical companies have also accelerated development of experimental vaccines, but they still will not be widely available in developing countries for some time.

To combat these problems, we propose a free smartphone app to detect COVID-19 from cough recordings, which would allow for mass-scale testing and could effectively stop the spread of the virus.

We decided to use offline edge prediction for our app for several reasons. Especially in developing countries, Internet connectivity is limited and people often face censoring. Data privacy regulations such as GDPR are now commonplace and on-device prediction will allow for diagnoses without personal information or health data crossing borders. Because our app will potentially serve billions of predictions daily, edge prediction is also more cost-effective, as maintaining and scaling cloud infrastructure to serve all of these predictions will be costly and difficult to maintain.


# System Design

In designing our system and pipeline, we first and foremost kept in mind that this pipeline would be running offline on edge devices in developing countries, including outdated phones with weak CPUs. We aimed for a pipeline that could efficiently process data, run a simple model, and return a prediction within a minute. To do this, we simplified our model, sacrificing some “expressiveness” in exchange for reduced complexity, but also through straightforward preprocessing of data.



*   how

For the frontend, we decided on a web app because it can be used in the browser, which is operating-system-agnostic; in comparison, apps may only run on certain operating systems. Our frontend is written in ReactJS + TypeScript, which is the industry standard for modern web design. It employs responsive web design principles to be compatible with a wide range of screen sizes and aspect ratios present on different devices. Internally, the frontend calls a TensorFlow.js (TFJS) model for inference

We chose to use the [TensorFlow.js](https://www.tensorflow.org/js) (TFJS) framework because it is supported for use with web browsers. The TFJS [Speech Command](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands) library provides a JavaScript implementation of the Fourier Transform (Browser FFT) to allow straightforward preprocessing of the raw audio files. We trained a vanilla TensorFlow model on background noise examples provided by the sample TFJS Speech Commands code, along with a dataset of thousands COVID-19 test result labeled coughs, so that our model could distinguish coughs from background noise. We then converted this trained model into the TFJS LayersModel format (with the model architecture as a JSON and weights in .bin files), so that we could integrate it into the front end JavaScript code for browser inference on-device.

Our system’s basic pipeline is as follows:



1. User opens our app
2. The TFJS models are downloaded from S3 onto the user’s device
3. Microphone detects noise from user
4. The Speech Commands library continuously preprocesses the audio by creating audio spectrograms
5. The spectrograms are run through the model
6. Only if the audio snippet is classified as a cough, the user will receive a prediction of whether they are COVID positive or negative

It is worth noting that model files are downloaded and loaded into memory only when the user first opens the app. After this, no Internet access is required and the system is able to make predictions offline.



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")




*   This section details the key components of the system, including, but not limited to, data pipelines, modeling, deployment, and UX.
*   If applicable, a diagram is included to illustrate the interplay between system components. Excalidraw is pretty awesome for sketches.
*   This section explains and justifies central design decisions, including that of which technologies the team chose to use to support their system.


# Machine Learning Component [Vivian/Solomon + Daniel]

The model that powers our application is based on the publicly available Tensorflow JS Speech Commands model. The model is intended to be used with the WebAudio API supported by all major browsers. This model expects, as input, audio data preprocessed with the browser FFT used in WebAudio’s GetFloatFrequencyData. The result of this FFT is a spectrogram which represents a sound wave as a linear combination of single-frequency components. The spectrogram, which can be thought of as a 2D image, is passed through a convolutional architecture to obtain logits which can be used in multiclass prediction. Specifically, this model has 13 layers with four pairs of Conv2D to MaxPooling layers, two dropout layers, a flatten layer, and two dense layers. 

Because training from scratch is expensive, we started with a model trained using the Tensorflow Speech Commands dataset [2], trained to recognize 20 common words such as the numbers “one” to “ten”, the four directions “left”, “right”, “up”, “down”, and basic commands like “stop” and “go”. We performed transfer learning on this model by removing the prediction layer and initializing a new one with the correct number of prediction classes. Afterwards, we fine-tuned the weights on our own cough datasets such as the CoughVID open source dataset. The CoughVID dataset provides 20,000 crowdsourced cough recordings from a plethora of different characteristics not limited to gender, geographic location, age, and COVID status. 

To ensure that data is preprocessed in the same way during training and testing, we use a custom preprocessing Tensorflow model which is trained to emulate the browser FFT that is performed by WebAudio, producing the same spectrogram as output. This browser-FFT emulating model is provided and maintained by Tensorflow Speech Commands. 

Originally, we had been training with Teachable Machine [2], but found that they did not support custom training pipelines. Creating our own training pipeline lets us use optimized custom architectures and hyperparameters discovered through ongoing research efforts, such as in Fakhry (2021) [3].   



*   This section explains the ML model(s) that powers the application, the data it’s trained on, and the iterative development of that model.
    *   coughvid
    *   Google speech commands library
    *   preprocess
    *   what inputs mean, model architecture, pretrained


# System Evaluation [Vivian/Solomon (ask Daniel)]

Offline evaluation was done on our model as a quick way to ensure our model was working correctly. This meant setting aside 30% of our data as test data. To monitor offline testing, we used Weights and Biases. As shown below, 50 epochs were sufficient to achieve convergence in training and validation accuracies, with corresponding decreasing losses. Here is an example of what we were logging: 

The rest of the evaluation was done through real world testing. The gold standard of testing would be large-scale, randomized clinical trials, with data collected from a variety of demographic groups and recording devices. We did not have the time and resources to do that in this class; instead we did some informal evaluations on our own team members. In addition, we also enlisted the help of volunteers around the globe. 

Anecdotally, the prediction was highly accurate on the group members, who are primarily Asian and all healthy. This remained true across a variety of devices such as smartphones, laptops, and tablets. 

The collection of external results was complicated by ethical considerations and lack of access to PCR tests to provide ground-truth labels. Nonetheless, we will note here two cases in Brazil. One individual was recovered, but previously was diagnosed with COVID-19; the model predicted that he had COVID-19. The other individual had COVID-19, but was predicted to be healthy. This illustrates the inherent challenge of translating models from development to production; model accuracy might be highly degraded due to distribution shift between the training and inference data. 



*   This section describes the team’s efforts to validate and evaluate their system performance as well as its limitations, both at instance-level and system-level.

        in our test model/split performed XX%, sensitivity, false negative is important


        system-level test => covid negative reported for self


        tested on apple and android smartphones, wear a mask


        unit testing


        tested in hospitals of COVID positives (if Amil can do it)

*   The results are included and presented in a clear and informative manner.

        graphs, weights and biases



# Application demonstration [Amil]

In the beginning stages of the design process, our product designer determined the appropriate target audience by conducting user interviews. We selected potential interviewee candidates based on certain demographic criteria such as being a citizen of selected Latin America countries or being tech-savvy and owning a cell phone.

After gathering target audience candidates from 7 Latin America countries as well as the U.S. and Pakistan, user interviews were conducted. The results from the interviews were then synthesized to create user personas. These personas helped us produce empathetic and user-centered designs throughout the whole design process.  

Once initial ideation and designs were completed, we conducted a series of prototype user tests in which the user is observed as they walk themselves through the app mockup. The data from each user test is then synthesized to design a new and improved iteration.

After numerous user tests and iterations and evolving over the past month, our designer created a finalized mockup of the demonstration application, choosing to build a web app because it works on all devices. Our customers are hospitals for initial testing, which will eventually expand to laypeople. We have made the instructions simple and easy to follow, as users just need to record their cough and they can immediately get their prediction.



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image2.png "image_tooltip")




<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.jpg "image_tooltip")



# Reflection [Everyone]

What worked? What didn’t work? 



1. **Teachable Machine**

    In the beginning, we had trouble dealing with some of the features in our model such as MFCC and mel-spectrogram as we tried to convert the model to TFJS. We further reached out to Pete Warden, expert of TinyML, about suggestions on how to keep everything on a smaller scale. This led to our first attempt of our system design using Teachable Machine, a web-based tool to create models. Teachable Machine uses Tensorflow.js to train models and then provides the code to integrate into JavaScript for a front end; it was simple and light. We soon discovered Teachable Machine was not that feasible however; it was difficult to feed in hundreds of examples for training because it required us to record your examples manually (therefore leading to poor performance), and we also wanted to know more what was actually going on in the training of the model and about the model itself. This ultimately forced us to train our own custom model. 

2. **Speech Commands Library**

    Tensorflow’s Speech Commands library provided a simple API to access a variety of important features like cutting the continuous audio stream into one-second snippets and performing FFT feature extraction to obtain spectrograms. The availability of pre-existing training pipelines as well as example applications using Speech Commands provided a strong foundation for us to adapt our own pipeline and frontend application. 


    One issue is that the Speech Commands library only processes .wav files. We ran into issues with converting between .wav files and .webm files, so had to run our own batch in order to deal with compatibility issues between different types of audio files, and consequently ended up with a smaller than ideal dataset. 

3. **Data**

    We had access to some more data such as the Coswara dataset, but did not have time to combine those with the CoughVid dataset we used.

4. **Team Dynamics**

    We compartmentalized responsibilities such that individual members were largely in charge of separate components of the system. Frequent communication was key to ensure that we all had a sense of the bigger picture. Our primary mode of communication was Slack which allowed us to constantly check up on each other’s progress.


    Overall, this experience taught us how to integrate the frontend and backend to build a Machine Learning system as well as utilize APIs and libraries to expedite the ML process.


If you had infinite time/resources, what would you improve? 



*   We would try to improve our model quality. [Research](https://arxiv.org/ftp/arxiv/papers/2103/2103.01806.pdf) suggests that 98% sensitivity is possible, which we currently fall below. 
*   Model development is limited by the lack of access to large-scale, demographically diverse, and accurately labelled datasets; we hope to remedy this in the future through improved data curation efforts. 
*   At the hardware level, it might be fruitful to calibrate individual microphones to reduce distribution shift between training and inference. 
    *   Model evaluation in production is complicated by the need to account for different microphone characteristics of different devices. 
*   Exploring the effect of compression, mp3, etc. may lead to interesting insights.
*   COVID-19 is not the only disease that affects one’s cough. We need to take into account various other diseases such as asthma, flu, and pneumonia, and distinguish between all of them (perhaps by using a multi-class classifier, or adding a class called “other”).
*   If possible, we would like to support upload of recorded coughs in common video formats like Mp4, Wav, and Webm. 
*   Embedded devices may be something to look into, alongside making the model even smaller.
*   This section provides a comprehensive post-mortem on the project, including - but not limited to - answering the following:
    *   What worked? (In terms of technology, design decisions, team dynamics, etc.).
    *   What didn’t work? What would you improve next time?
    *   If given unlimited time and resources, what would you add to your application?
    *   If you have plans to move forward with this application, what are they? (We’re always excited to see how students use the tools they’ve learned in this class to pursue topics they’re excited about!)


# Broader Impacts [Amil]

Our app is intended to be used by people in developing countries who need an anonymous solution for testing anytime, or by anyone in a community at risk of COVID-19. However, we have identified some unintended uses of our app.

Because we intend to share our technology for free and because the algorithm runs on-device, competitors will be able to create copies of our app by stealing our algorithm, and may block access to the app or sell the app for profit when it is intended to be provided for free. To prevent this, we will open source our technology under terms requiring attribution to Virufy and prohibiting charging the users for use of the algorithm.

Another risk is that people may begin to ignore medical advice and believe only in the algorithm and might use the results in place of an actual diagnostic test. This is very risky because if the algorithm mispredicts, Virufy will be held liable, and the spread of COVID-19 may increase because COVID-19 positive people are confident to socialize with their false negative test results. To mitigate this, we intend to add disclaimers that our app is a pre-screening tool that should be used in conjunction with medical providers’ input. Additionally, we will work closely with public health authorities to clinically validate our algorithm and ensure it is safe for usage.

People may also start testing the algorithm with irrelevant recordings of random noises such as talking. To address this, we have equipped our algorithm with a cough detection pre-check layer to prevent any non-cough noises from being classified.

Finally, people especially in poorer contexts may share the same smartphones for several users, which can increase the likelihood of spreading COVID-19. Thus, our instructions clearly state that users must disinfect their device and keep 20 feet away from others while recording.



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")



# References [Amil]

[1] Tensorflow Speech Commands dataset, [https://arxiv.org/pdf/1804.03209.pdf](https://arxiv.org/pdf/1804.03209.pdf)

[2] Teachable Machine, [https://teachablemachine.withgoogle.com/](https://teachablemachine.withgoogle.com/)

[3] Virufy: A Multi-Branch Deep LearningNetwork for Automated Detection ofCOVID-19 [https://arxiv.org/ftp/arxiv/papers/2103/2103.01806.pdf](https://arxiv.org/ftp/arxiv/papers/2103/2103.01806.pdf)

We’re extremely grateful to [Pete Warden](https://petewarden.com/), [Jason Mayes](http://www.jasonmayes.com/), and [Tiezhen Wang](https://www.linkedin.com/in/tiezhen/) from Google’s Tensorflow JavaScript team for their kind guidance on TinyML concepts and usage of the [speech_commands library](https://www.tensorflow.org/datasets/catalog/speech_commands), both in class lecture and during the few weeks of our development.

[Jonatan Jaskilioff](https://www.linkedin.com/in/jonatan-jaskilioff-77075340/) and the team at [XOOR](https://xoor.io/) were very gracious to lend their support and guidance in integrating our JavaScript code into the [progressive web app](https://virufy.org/demo) they had built pro bono for Virufy.

We are also indebted to [Amil Khanzada](http://www.amilkhanzada.com/) and the broader [Virufy](http://virufy.org/) team for guiding us on the real-world applicability and challenges of our edge device prediction project. We leveraged their deep insights from their members distributed across 20 developing countries in formulating our problem statement. Additionally, we took advantage of the [demo app](https://virufy.org/demo) that they had built prior based on intentions for real-life usage, along with their prior [research findings](https://virufy.org/paper) and [open source code](https://github.com/virufy/covid) for our model training.

In preparing our final report, we are grateful to [Colleen Wang](https://www.linkedin.com/in/colleen-wang-59a091205/) for her kind support in editing the content of our post, Virufy lead UX designer [Maisie Mora](https://www.linkedin.com/in/maisiemora/) for helping explain her design thinking and process in the application demonstration section, [Saad Aslam](https://www.linkedin.com/in/saslam23/) for his kind support in converting our blog post to a nicely formatted HTML page, and [Victor Wang](https://www.linkedin.com/in/executivestanfordvictorwang/) for his kind support in reviewing our post for high-level audiences. 

Finally, we cannot forget the great lessons and close guidance from Professor [Chip Huyen](https://huyenchip.com/) and TA [Michael Cooper](https://michaeljohncooper.com/) who helped us open our eyes to production machine learning and formulate our problem to be attainable within the short 2 month course quarter.

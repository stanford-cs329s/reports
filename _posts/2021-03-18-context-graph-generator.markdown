---
layout: post
title: Building a Context Graph Generator
date: 2021-03-18 20:32:20 +0700
description: We developed a context graph generator capable of providing visual (graphical) summaries of input text, highlighting salient concepts and their connections.
img: # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Graph-ML, NLP, Streamlit]
comments: true
---
 
### The Team
- Manan Shah
- Lauren Zhu
- Ella Hofmann-Coyle
- Blake Pagon

### Problem Definition
<!-- - This section explains the problem the team is solving, discusses related work, and proposes and justifies their solution.
- Explain the main contributions of your project (e.g. improving model performance, improving latency, getting it to work on edge, novel use case, etc.) -->

Get this—55% of users read online articles for less than 15 seconds. The general problem of understanding large spans of content is painstaking, with no efficient solution. 

Current attempts rely on abstractive text summarization, which just shortens text to its most relevant sentences and often obscures core components that make writing individualistic and meaningful. Other methods offer a smart search on articles, a sort of smart command F. But that forces the user to look for bits and pieces of content at a time. They also have to know beforehand what they’re looking for. 

What if, rather than engaging in this lengthy process of understanding content manually, people can leverage an algorithm that computes concept graphs over text? These graphs provide the first-ever standardized, structured, and interpretable medium to understand and draw connections between large spans of content. We create an interface where users are able to visualize and interact with the key concepts of swaths of text in a meaningful and unique way. Users can even compare and contrast concept graphs between multiple articles with intersecting concepts, to further their understanding.

So with users spending less time on individual websites, we make it easier for them to absorb the key components of the content in an interactive graph representation. Specifically, we let readers
- Grasp the main concepts presented in an article
- Understand the connections between topics
- Draw insights across multiple articles fluidly and intuitively


### System Design
<!---
- This section details the key components of the system, including, but not limited to, data pipelines, modeling, deployment, and UX.
- If applicable, a diagram is included to illustrate the interplay between system components. [Excalidraw](https://excalidraw.com/) is pretty awesome for sketches.
- This section explains and justifies central design decisions, including that of which technologies the team chose to use to support their system.
-->
#### Graph Generation

<img style="float: right; width:280px;" src="../assets/img/context-graph-generator/image_1.png">

With the goal of making information digestion faster, we aimed to design a concept graph that not only includes relevant information on a text’s main concepts, but also visually shows how topics relate to one another. To achieve this we designed our graph outputs such that nodes represent article concepts, which are determined programmatically, and edges represent links between the concepts. 

To do this we utilize streamlit for the front end interface and graph using our custom version of the streamlit graphing libraries. The resulting graphs are interactive; users can move the graph and individual nodes, and can also click a node to receive a pop up that shows all context in which the corresponding topic exists in the text. 

Additionally, we provide users with a threshold slider that allows users to decide how many nodes/connections they want their graph to provide in order to better meet their needs. How this works is that connections between nodes are determined by calculating a similarity score between the nodes (via cosine similarity on the word embeddings), and a connection is drawn between two topics if the score is above the threshold from the slider. This means that as the slider moves further to the right, the lower threshold makes the graph generate more nodes and thus the resulting graph is more dense. 

#### Working with Multiple Graphs

In addition to generating graphs from a single text source, users can combine graphs they generate to see how concepts from several articles relate. In the below example, we see how two related articles interact when graphed together. Here we have one from the Bitcoin Wiki the other on Decentralized Finance. We can distill from a quick glance that both articles talk about networking, bitcoin, privacy, block chain and currency, but diverge as the Bitcoin article focuses on the system specification of Bitcoin and the Decentralized Finance article talks more about impacts of Bitcoin on markets. The multi-graph option allows users to not only assess the contents of several articles all at once with a single glance, but can also reveal insights on a topic through showing the interconnections of the two sources, providing a holistic view of any topic a user wants to delve into.

<img style="display: block; margin-left: auto; margin-right: auto; width:600px;" src="../assets/img/context-graph-generator/image_2.png">

#### Visual Embedding Generation

An additional feature our system provides users is a tool to plot topics in 2D and 3D space in order to provide further insights and analysis on article topics. We utilize the Plotly library in order to make these plots interactive! The embedding tools simply takes the embedding that corresponds to each topic node in our graph and projects it into 2D space. Topic clustering indicates high similarity or strong relationships between those topics. Large distances between topic points in 2D indicate that topics are more dissimilar. We also give users the ability to upgrade their 2D plots to 3D, if they're feeling especially adventurous.

#### Deployment and Caching

We deployed our app on Google Cloud Platform (GCP) via Docker. We sent a cloud built docker image to Google Cloud, and set up a powerful VM that accessed that launched the app from that Docker image. For any local updates to the application, redeploying was quite simple—We just had to rebuild the image using Google Cloud Build, and point the VM to the updated image. 

To speed up performance of our app, we cache graphs globally, that way if you are trying to graph an article about Bitcoin and another user recently generated the same graph, our system will quickly serve the cached graph! To do this we make use of streamlit’s caching features and update each user’s session state to remember which graphs a user generated in order to avoid user’s queries from interfering with each other’s experience. 

### Our Backend: Machine Learning for Concept Extraction and Graph Generation
<!---
- This section explains the ML model(s) that powers the application, the data it’s trained on, and the iterative development of that model.
-->

In order to generate the concept graph, we segmented our pipeline into four key stages, as in Figure 1. In particular, users are allowed to provide either custom input consisting of arbitrarily-formatted text or a web URL, which we parse and extract relevant textual information from. We next generate concepts from text using numerous concept extraction techniques, including TF-IDF and PMI-based ranking over extracted n-grams. The resulting combined topics are culled to the most relevant ones, which are subsequently contextualized by sentences that contain the topics. Finally, each topic is embedded according to its relevant context, and these embeddings are used to compute (cosine) similarities and generate the visualizations produced as output for the user. Our primary machine intelligence pipelines are introduced in (1) our TF-IDF concept extraction of relevant topics from the input text and (2) our generation of BERT embeddings of each topic using the contextual information of the topic within the input text. 

<img style="display: block; margin-left: auto; margin-right: auto; width:700px;" src="../assets/img/context-graph-generator/image_3.png">

Pipeline Illustration: A diagram of our text-to-graph pipeline, which uses machine intelligence models to extract concepts from an arbitrary input span of text.

We began our concept extraction pipeline by simply using the most frequent unigrams and bigrams present in the input text, but we soon realized that doing so populated our graph with numerous meaningless words that had little to do with the article and instead represented common terms and phrases broadly used in the English language. Although taking stopwords into account and ranking bigrams by their pointwise mutual information partially resolved this issue, we were left unable to consistently obtain concepts that accurately represented the input document. In order to properly resolve this issue, we pre-processed a large Wikipedia dataset consisting of 6 million examples to extract “inverse document frequencies” for common unigrams, bigrams, and trigrams. Each topic was ranked according to its term frequency-inverse document frequency ratio, representing the uniqueness of the term to the given article compared to the overall frequency of the term in a representative sample of English text. Doing so allowed us to properly select topics that were unique to the input documents, significantly improving our graph quality.

To embed each concept contextually, we began by using pre-trained embeddings from GloVe and word2vec, which each embed words using neural networks trained on context windows that place similar words close to each other in embedding space but fail to consider context when making predictions. This limitation was particularly problematic for our use-case, as using pre-trained context-independent word embeddings would result in identical graphs being produced for each set of concepts as opposed to a graph that would be meaningful and specific for each input article. We verified this hypothesis by asking users to evaluate the quality of the generated graphs, with the primary feedback being that the graph represented abstract connections between concepts as opposed to the connections that were represented in the provided text. In order to resolve this issue and better generate contextually-relevant graphs, we introduced a BERT embedding model that embedded each concept along with its surrounding context, producing an embedding for each concept that was influenced by the article it was present in. Our BERT model was pre-trained on BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia (excluding lists, tables and headers). We used embeddings from the final layer of the BERT model, averaged across all WordPiece-split tokens describing the input concept, to generate our final 1024-dimensional embeddings for each concept. In order to improve the efficiency of this overall process and guarantee graph generation completes in under 1 minute for user inputs of reasonable length, we implemented caching mechanisms to ensure that identical queries would have their associated embeddings and adjacency matrices cached for future use.

### System Evaluation
- This section describes the team’s efforts to validate and evaluate their system performance as well as its limitations, both at instance-level and system-level.
- The results are included and presented in a clear and informative manner.

### Application Demostration
<!---
- This section includes visuals (screengrab, embedded video link) showcasing the main feature set of the application.
- The section also includes brief justifications of core interface decisions made by the team (e.g. why did the team feel that a Web Application interface would be superior to an API interface given the context of their problem?).
- Instructions on how to use the application.
-->
Interested users may visit our application at [this URL](http://104.199.124.26/), where they are presented with a simple and intuitive interface that allows them to (a) input custom text to graph, (b) input a web URL to graph, or (c) generate a combined graph from two of their previously saved graphs. Upon entering custom text or a web URL, users are shown a progress bar estimating the time of graph generation (or an instant graph if the query has been cached from previous uses of the website). Generated graphs are interactive, allowing users to click on nodes to see the context in which they appear in the input document. Other modes of visualization are additionally presented, including a 2D and 3D PCA-based embedding of the concepts to provide a different perspective of relationships between the concepts. Users are further provided with an option to save graphs locally (to their local browser cache), and subsequently combine them to link concepts together across numerous distinct documents.

Our team chose to use a web interface as it proved to be the most intuitive and straightforward way for users to provide custom input and interact with the produced graphs. We implemented our own customizations to the streamlit default graphing library (in [this fork](https://github.com/mananshah99/streamlit-agraph)) to enable enhanced interactivity, and we employed streamilit to ensure a seamless development process between our python backend and the frontend interface. 

<iframe width="955" height="512" src="https://www.youtube.com/embed/bcGmY7XTokM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Reflection
<!--
- This section provides a comprehensive post-mortem on the project, including - but not limited to - answering the following:
	- What worked? (In terms of technology, design decisions, team dynamics, etc.).
	- What didn’t work? What would you improve next time?
	- If given unlimited time and resources, what would you add to your application?
	- If you have plans to move forward with this application, what are they? (We’re always excited to see how students use the tools they’ve learned in this class to pursue topics they’re excited about!)
-->
#### What worked well?
We had such a rewarding and exciting experience this quarter building out this application. From day one, we were all sold on the context graph idea and committed a lot of time and energy into it. We are so happy with the outcome that we want to continue working on it next quarter. We will soon reach out to some of the judges that were present at the demo but didn’t make it to our room.

While nontrivial, building out the application was a smooth process for several reasons: technical decisions, use of Streamlit, great camaraderie. Let’s break these down.

Our topic retrieval process is quite simple, with the use of highest frequency n-grams, with n={1,2,3}. TF-IDF was a good addition to topic filtering (the graphs were more robust as a result), but because it was slower we added it as a checkbox option to the user. Sentence/context retrieval was also quite simple, with the use of a regex for the n-gram. We then just had to shape the topics and contexts correctly and after passing them through BERT, compute cosine similarities. For displaying graphs, we found a Streamlit component called Agraph. While it had all the basic functionality we needed, there were things we wanted to add on top of it (e.g. clicking on nodes to display context), which required forking the repo and making custom changes on our end.

Due to the nature of our project, it was pretty feasible to build out the MVP on Streamlit and iterate by small performance improvements and new features. This made individual contributions easy to line up with git issues and to execute on different branches. It also helped that we have an incredible camaraderie already, as we all met in Stanford’s study abroad program in Florence in 2019. 

#### What didn't work as well?
To be honest, nothing crazy here. We had some obscure bugs from BERT embeddings that would occur rarely but at random, as well as graph generation bugs if inputs were too small. We got around them with try/catch blocks, but could have looked into them with a little more attention.

#### If we had unlimited time & unlimited resources...
Among the four of us, we made our best guesses as to what the best features would be for our application. Of course if time permitted, we could conduct serious user research about what people are looking for, and we could build exactly that. But apart from that, there are actual action items moving forward, discussed below. 

We wanted to create an accessible tutorial or perhaps some guides either on the website or an accessible location. This may actually no longer be necessary because we can point to the tutorial provided in this blog (see Application Demonstration). We saw in many cases that without any context of what our application does, the user may not know what it is for or how they could get the most out of it. 

On top of this, future work includes adding a better URL (contextgraph.io?), including a Chrome extension, and making a pull request to the Streamlit Agraph component with our added functionality—in theory we could then deploy this for free via Streamlit.

### Broader Impacts
- This section discusses intended uses of your application - and possible unintended uses, and the associated harms.
- This section reflects upon the design decisions that the team undertook to mitigate harms associated with unintended use of the system.

### Contributions
- Write down a brief summary of the individual contributions of each of the team members.

### References
- Correctly and comprehensively cites any sources used to assist the development of the application, including, but not limited to, papers, tutorials, and interviews/conversations.
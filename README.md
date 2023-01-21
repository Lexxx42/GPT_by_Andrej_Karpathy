# GPT_by_Andrej_Karpathy
Let's build GPT: from scratch, in code, spelled out.
Link to source material: https://youtu.be/kCc8FmEb1nY
GitHub: https://github.com/karpathy/ng-video-lecture

![Nodes](https://www.techiedelight.com/wp-content/uploads/Eulerian-path-for-directed-graphs.png "nodes")


## Suggested exercises:
- EX1: The n-dimensional tensor mastery challenge: Combine the `Head` and `MultiHeadAttention` into one class that processes all the heads in parallel, treating the heads as another batch dimension (answer is in nanoGPT).
- EX2: Train the GPT on your own dataset of choice! What other data could be fun to blabber on about? (A fun suggestion if you like: train on all the possible 3-digit addition problems and predict the sum in the reverse order. Does your Transformer learn the correct addition algorithm? Does it correctly generalize to the validation set?). 
- EX3: Find a dataset that is very large, so large that you can't see a gap between train and val loss. Pretrain the transformer on this data, then initialize with that model and finetune it on tiny shakespeare with a smaller number of steps and lower learning rate. Can you obtain a lower validation loss by the use of pretraining?
- EX4: Read some transformer papers and implement one additional feature or change that people seem to use. Does it improve the performance of your GPT?

![The transformer model from "Attention is all you need", Viswani, et. al.](https://www.researchgate.net/profile/Dennis-Gannon-2/publication/339390384/figure/fig1/AS:860759328321536@1582232424168/The-transformer-model-from-Attention-is-all-you-need-Viswani-et-al.jpg "The transformer model (from \"Attention is all you need\", Viswani, et. al.)")

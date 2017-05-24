Notes on the Nematus architecture and code...

In out tiny_sample:

```
VOCAB_SIZE = 90000
dim_words = 500
n_words = 90000
n_words_src = 90000
dim = 1024
```

Initializes numpy arrays for all the parameter matrices, with random
values.  At this point these are actual numpy arrays with an observed
shape.  These later get fed into theno shared variables where they are
harder to observe.

The paper omits biases but they are shown below.

Each GRU has 6 weight matrices plus bias for a total of 9 parameters
per GRU.  The code concatenates the Wr and Wz matrices into one W
matrix.  It does similar for Ur and Uz into U and bz, br into b.  So
we end up with 6 named parameters per GRU in the code.

Below, simple vectors are shown with dimensions (N, ), because these
are numpy shapes, which are python tuples.  A tuple with one element
is written with an empty second element.

```
embedding
  Wemb                       (90000, 500)          source word embeddings
  Wemb_dec                   (90000. 500)          target word embeddings (in decoder)
```

```
encoder: bidirectional RNN
  forward RNN
    encoder_W                (500, 2048)
    encoder_b                (2048, )
    encoder_U                (1024, 2048)
    encoder_Wx               (500, 1024)
    encoder_bx               (1024, )
    encoder_Ux               (1024, 1024)

  reverse RNN
    encoder_r_W              (500, 2048)
    encoder_r_b              (2048,)
    encoder_r_U              (1024, 2048)
    encoder_r_Wx             (500, 1024)
    encoder_r_bx             (1024, )
    encoder_r_Ux             (1024, 1024)
```

```
init_state, init_cell
  ff_state_W                 (2048, 1024)           Winit
  ff_state_b                 (1024, )
```

```
decoder
  GRU1
    decoder_W                (500, 2048)            W'r W'z
    decoder_b                (2048, )               (combined biases for reset and update gates)
    decoder_U                (1024, 2048)           U'r U'z
    decoder_Wx               (500, 1024)            W'
    decoder_Ux               (1024, 1024)           U'
    decoder_bx               (1024, )

  GRU2
    decoder_U_nl             (1024, 2048)
    decoder_b_nl             (2048, )
    decoder_Ux_nl            (1024, 1024)
    decoder_bx_nl            (1024, )
    decoder_Wc               (2048, 2048)
    decoder_Wcx              (2048, 1024)

  attention
    decoder_W_comb_att       (1024, 2048)           Ua
    decoder_Wc_att           (2048, 2048)           Wa
    decoder_b_att            (2048, )               
    decoder_U_att            (2048, 1)              va
    decoder_c_tt             (1, )                  ?    ask Scott?  did we decide this is like a bias?
```

```
readout
  from GRU2
    ff_logit_lstm_W          (1024, 500)            Wt1
    ff_logit_lstm_b          (500, )
  prev word
    ff_logit_prev_W          (500, 500)             Wt2
    ff_logit_prev_b          (500, )
  context
    ff_logit_ctx_W           (2048, 500)            Wt3
    ff_logit_ctx_b           (500, )
  output 1-of-k
    ff_logit_W               (500, 90000)           Wo
    ff_logit_b               (90000, )
```

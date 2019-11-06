# Voice-Conversion-StarGAN
### Paper and Dataset

**Paper：** https://arxiv.org/abs/1806.02169 

**Dataset：**[VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211)

*Note that the model architecture is a little different from that of the original paper.*

<br/>

### Dependencies

* Python 3.7
* Pytorch 1.2.0
* pyworld
* librosa
* numpy 1.17.2

<br/>

### File Structure

```bash
|--convert.py
|--data_loader.py
|--logger.py
|--main.py
|--model.py
|--preprocess.py
|--solver.py
|--utils.py
|--data--|vcc2016_training
       --|evaluation_all
```



### Usage

#### Preprocess

```bash
python preprocess.py 
```

*Note: It is much faster to use the multiprocess than not.*

<br/>

#### Train

```bash
python main.py
```

<br/>

#### Inference

For example: restore model at step 200000 and specify the source speaker and target speaker to `SF1` and `TM1`, respectively.

```bash
python convert.py --resume_iters 200000 --src_spk p262 --trg_spk p272
```

<br/>

### To-Do list

- [x] Provide some samples


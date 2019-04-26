# https://medium.com/looka-engineering/how-to-use-tensorboard-with-pytorch-in-google-colab-1f76a938bc34
"""
#Install
pip install tensorboardX

#Tensorboard Run
tensorboard --logdir ./log --host 0.0.0.0 --port 6006

#using ngrok/tunell (not required)
pip install ngrok
./ngrok http 6006

#Google Colab Run
!pip install tensorboardX
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
#End Google Colab
"""


from tensorboardX import SummaryWriter
from fastai.callback import Callback
from time import gmtime, strftime

class TensorBoardMonitor(Callback):
    """
    Base class for callbacks that want to record values, dynamically change learner params, etc.
    """
    def _mark_text(md):
      return str(md).replace('\n','  \n')  
  

    def __init__(self, 
                 learn, 
                 comment='TensorBoard Monitor', 
                 log_dir='./log',
                 save_model=True,
                 hist_freq=1,
                 vision_top_losses_freq=1):
        """
        general callback for fast.ai used in TensorBoard
        Args:
          learner: name of the learner
          log_dir: path for the log dir run from TensorBoard
          save_model: save best model after each epoch
          hist_freq: how often save histogram for model parameters 0 (None)
        """      
        super().__init__()
        self.learn = learn
        self.learn.callback_fns.append(self)
        self.log_dir = Path(log_dir)
        self.save_model = save_model
        self.hist_freq = hist_freq
        self.vision_top_losses_freq = vision_top_losses_freq
        
        
        self.metrics_names = ["validation_loss"]
        self.metrics_names += [m.__name__ for m in learn.metrics]

    
    def __call__(self,learn):
        self.learn = learn
        return self
      
    def write_summary(self):
        self.best_met = 0
        self.run_name = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.run_dir = self.log_dir/self.run_name
        self.writer = SummaryWriter(
              log_dir=str(self.run_dir),
              comment='TensorBoard Monitor', 
            )

        dummy_input = torch.zeros(learn.data.one_batch()[0].shape).cuda()
        self.writer.add_graph(self.learn.model, dummy_input)
        self.writer.add_text('Summary', TensorBoardMonitor._mark_text(learn.summary()))
        self.writer.add_text('Model', TensorBoardMonitor._mark_text(learn.model))

    def write_params_histogram(self,n_iter):
       for name, param in self.learn.model.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
            
    def write_metrics(self, metrics, n_iter):
        for val, name in zip(metrics, self.metrics_names):
            self.writer.add_scalar(name, val, n_iter)        
            
    def write_model(self, metric):
        print('Saved new model to: ', 
              self.learn.save(self.run_name, return_path=True), 
              ' (', metric,')')    
            
      
    def on_train_begin(self, **kwargs):
        self.write_summary()
        self.writer.add_text('Hyperparameters', 
                            f'lr: {self.learn.opt.lr}, mom: {self.learn.opt.mom}, wd: {self.learn.opt.wd}, beta: {self.learn.opt.beta}')
        

    def on_batch_end(self, **kwargs):
        # Add single scalara for num_batch
        self.trn_loss = kwargs['last_loss']
        num_batch = kwargs['num_batch']
        self.writer.add_scalar('trn_loss_batch', self.trn_loss, num_batch)

    def on_epoch_end(self, **kwargs):
        metrics = kwargs['last_metrics']
        epoch = kwargs['epoch']
        trn_loss = kwargs['smooth_loss']
        
        #Add loss
        self.writer.add_scalar('trn_loss', trn_loss, epoch)
        
        #Save all histograms
        if self.hist_freq>0 and (epoch % self.hist_freq)==0:
          self.write_params_histogram(epoch)
        
        #Add all metrics
        self.write_metrics(metrics, epoch)

        # Save best model, in the file
        m = metrics[1]
        if m > self.best_met and self.save_model:
            self.best_met = m
            self.write_model(m)
            
        #Add figure
        if (self.vision_top_losses_freq>0 
          and (epoch % self.vision_top_losses_freq)==0
          and isinstance(learn.data.x,ImageList)
          and isinstance(learn.data.y,CategoryList)):
            interp = ClassificationInterpretation.from_learner(learn);
            fig = interp.plot_top_losses(9, 
                                         figsize=(15,11), 
                                         heatmap=True,
                                         return_fig=True);
            self.writer.add_figure('top losses', fig)

    def on_train_end(self, **kwargs):
        self.writer.add_text('Total Epochs', str(kwargs['epoch']))
        self.writer.close()


#     def on_epoch_begin(self, **kwargs:Any)->None:
#         "At the beginning of each epoch."
#         pass
#     def on_batch_begin(self, **kwargs:Any)->None:
#         "Set HP before the output and loss are computed."
#         pass
#     def on_loss_begin(self, **kwargs:Any)->None:
#         "Called after forward pass but before loss has been computed."
#         pass
#     def on_backward_begin(self, **kwargs:Any)->None:
#         "Called after the forward pass and the loss has been computed, but before backprop."
#         pass
#     def on_backward_end(self, **kwargs:Any)->None:
#         "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
#         pass
#     def on_step_end(self, **kwargs:Any)->None:
#         "Called after the step of the optimizer but before the gradients are zeroed."
#         pass
#     def on_batch_end(self, **kwargs:Any)->None:
#         "Called at the end of the batch."
#         pass
#     def on_epoch_end(self, **kwargs:Any)->None:
#         "Called at the end of an epoch."
#         pass
#     def on_train_end(self, **kwargs:Any)->None:
#         "Useful for cleaning up things and saving files/models."
#         pass
#     def jump_to_epoch(self, epoch)->None:
#         "To resume training at `epoch` directly."
#         pass        
    
    def __repr__(self):
        return f'(TensorBoard callback on the logdir: {self.run_dir})'



from fastai.tabular import *
import neptune
import datetime


class NeptuneMonitor(Callback):
    def __init__(self, 
                 learn,
                 api_token, 
                 project, 
                 tag='fastai', 
                 prefix=''):
        super().__init__()
        self.learn = learn
        self.learn.callback_fns.append(self)
        
        self.project = project
        self.tag = tag
        
        neptune.init(
            api_token = api_token,
            project_qualified_name=project)

        self._prefix = prefix
        self._run_number = 0
        
    def start(self,name):
        self._exp_uuid = name
        self._exp = neptune.create_experiment(name=self._exp_uuid)
        self._exp.append_tag(self.tag)
        self._exp.append_tag(self._exp_uuid)

        print('Create new experemiment: ', self._exp_uuid)
        
    def stop(self):  
      self._exp.stop()
      self._exp = None
      
    def set_property(self,key,value): self._exp.set_property(key,value)
    def send_metric(self,key,value): self._exp.send_metric(key,value)
    def send_text(self,key,value): self._exp.send_text(key,value)
    def send_image(self,key, PIL_image): self._exp.send_image(key,PIL_image)
    def send_artifact(self,path): self._exp.send_artifact(path)
    def append_tag(self,tag): self._exp.append_tag(tag)
        
    def _get_name(self,prefix=''):
        return prefix + '-' + self._exp_uuid + '-' + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        
    def send_df(self, df, prefix='submission', add_id = True, index=False):
        path = './df/' + prefix + '.csv'
        if add_id:
            path = './df/' + self._get_name(prefix) + '.csv'
        df.to_csv(path,index=index)
        self.send_artifact(path)
   
    def send_description(self,name,text, ext='.txt'):
        with open('./tmp/' + name+ext,"w") as f:
            f.write(text)
        self.send_artifact(name+ext)


    # FAST.AI PART for 
    def export(self):
        path = './models/'
        name = self._get_name();
        callback = self.learn.callback_fns
        self.learn.callback_fns = []
        self.learn.export(path+name)
        self.learn.callback_fns = callback
        
        print('Saved to:', path+name)
        return path, name
        
    def send_model(self):
        if self.learn is None:
            return
        path, name =  self.export()
        self.send_artifact(path+name)
        
        return path,name
        
        
    def __call__(self,learn):
        self.learn = learn
        return self

    def on_train_begin(self, **kwargs):
        self._run_number += 1
        self._exp.set_property('lr', str(self._run_number))

        with open("model.txt","w") as f:
          f.write(str(self.learn.model))
        self._exp.send_artifact('./model.txt')
        
        with open("opt.txt","w") as f:
          f.write(str(self.learn.opt))
        self._exp.send_artifact('./opt.txt')
        
        #self._exp.send_text('summary', str(self.learn.summary()))
        #self._exp.send_text('opt', str(self.learn.opt))
        self._exp.set_property('lr', str(self.learn.opt.lr))
        self._exp.set_property('mom', str(self.learn.opt.mom))
        self._exp.set_property('wd', str(self.learn.opt.wd))
        self._exp.set_property('beta', str(self.learn.opt.beta))
      
        
    def on_epoch_end(self, **kwargs):
        self._exp.send_metric(self._prefix + 'train_smooth_loss', float(kwargs['smooth_loss']))
        
        metric_values = kwargs['last_metrics']
        metric_names = ['valid_last_loss'] + kwargs['metrics']
        
        for metric_value, metric_name in zip(metric_values, metric_names):
            metric_name = getattr(metric_name, '__name__', metric_name)
            self._exp.send_metric(self._prefix + metric_name, float(metric_value))

    def on_batch_end(self, **kwargs):
        self._exp.send_metric('{}last_loss'.format(self._prefix), float(kwargs['last_loss']))
        
        
    def on_train_end(self,**kwargs):
        pass
    
    def __repr__(self):
        return f'<Neptune.ml for project: "{self.project}">' 

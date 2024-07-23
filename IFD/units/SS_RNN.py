import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import pdb


class SsCell(keras.layers.Layer):
#This is a class defining a non-linear state space cell. It only processes one time sample; thus, the input
#is expected to be of shape (batch_size,features)
  #TODO: add a line of code such that the initializer results in random (seudo random) weights every time it is called
  _weight_initializer=keras.initializers.RandomNormal() #std=0.05

  def __init__(self,num_states=50,output_dimension=10,state_nn_layers=None):
    super().__init__()
    self.num_states=num_states
    self.output_dim=output_dimension
    self.A=tf.Variable(initial_value=self._weight_initializer(shape=[num_states,num_states],dtype='float32'),name='A',trainable=True)
    self.b_state1=tf.Variable(initial_value=self._weight_initializer(shape=[num_states,],dtype='float32'),name='b_h1',trainable=True)
    self.b_state2=tf.Variable(initial_value=self._weight_initializer(shape=[num_states,], dtype='float32'),name='b_h2',trainable=True)

    #the B-matrix is initialized in the build method since it's shape depends on the input size
    #self.dummy_var=tf.zeros([self.num_states,]) # This variable will be passed -as a placeholder- to the MultivariateNormalTril layer that models the noise
    self.mdl_is=None
    self.mdl_o=None
    self.B=None


    #the following enables the user to control the architecture of the state-transition neural network. Note that the input and output layers
    #of the network are fixed
    if state_nn_layers is None:
        self.state_nn_layers=[keras.layers.Dense(64,activation='relu',name='l1-is',trainable=True),#tfp.layers.DenseVariational(64,make_posterior_fn=self._posterior_fn,make_prior_fn=self._prior_fn,kl_weight=0.1,activation='relu',name='l1-is',trainable=True),
               keras.layers.Dense(32,activation='relu',name='l2-is',trainable=True),
               keras.layers.Dense(16,activation='relu',name='l3-is',trainable=True)]
    else:
        self.state_nn_layers=state_nn_layers

  # def _prior_fn(self,kernel_size,bias_size,dtype=None):
  #   n= kernel_size+bias_size
  #   return tf.keras.Sequential([
  #     tfp.layers.VariableLayer(n, dtype=dtype),
  #     tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
  #         tfp.distributions.Normal(loc=t, scale=1),
  #         reinterpreted_batch_ndims=1)),
  # ])
  #
  # def _posterior_fn(self,kernel_size,bias_size,dtype=None): #the posterior and prior functions are based on tensorflow documentation <NOTE>: reinterpreted_batch_ndims is not fully clear (I believe it answers your question about how the kernel is modeled as a RV)
  #     n= kernel_size+bias_size
  #     c = tf.math.log(tf.math.expm1(1.))
  #     return tf.keras.Sequential([tfp.layers.VariableLayer(2*n,dtype=dtype),
  #                                                                tfp.layers.DistributionLambda(make_distribution_fn= lambda t: tfp.distributions.Independent(tfp.distributions.Normal(loc=t[...,:n],scale=1e-5 + tf.nn.softplus(c + t[..., n:]),
  #                                                                ),reinterpreted_batch_ndims=1))
  #                                                                ])

  def get_mdl(self,state_dim):
    layers_is=[keras.Input(shape=[state_dim,],name='input_of_is_mdl'),
               *self.state_nn_layers,
               keras.layers.Dense(state_dim,activation='linear',name='o-is')]

    mdl_is = keras.Sequential(layers_is)
    c=tf.math.expm1(1.)


    layers_o=[keras.Input(shape=[state_dim,],name='input_of_is_mdl'),
              #keras.layers.Dense(64,activation='sigmoid',trainable=True),
              keras.layers.Dense(2*self.output_dim,activation='linear',trainable=True),
              #TODO: I removed the tfp.layers.DistributionLambda since it is not
              tfp.layers.DistributionLambda(lambda t: tfp.distributions.MultivariateNormalDiag(loc=t[...,:self.output_dim],scale_diag=1e-5 + tf.nn.sigmoid(c + t[..., self.output_dim:])))  #Something is wrong here
              #I need to use a tfp.distribution.MultivariateNormalDiag or Tril (Done)
              ]
    mdl_o=keras.Sequential(layers_o)
    return mdl_is,mdl_o

  def build(self,inputs_shape):
    #two inputs are expected (input,state). Each has the shape of [batch_size,features]
    input_shape=inputs_shape[0] #[batch_size,input_features]
    state_shape=inputs_shape[1] ##[batch_size,state_features]
    self.B=tf.Variable(initial_value=self._weight_initializer(shape=[input_shape[-1],self.num_states],dtype='float32'),name='B') #B is originally n by m where n is the number of states and the m is the dimension of the input
    #but since for a batch calculation it will be transposed it is better (computationally) to define it like this
    self.mdl_is,self.mdl_o=self.get_mdl(state_shape[-1])

 #@tf.function()
  def call(self,inputs):
    input,state=inputs
    batch_size=tf.shape(input)[0]
    temp=tf.linalg.matmul(state,self.A)+tf.linalg.matmul(input,self.B)+self.b_state1
    state=self.mdl_is(temp)+self.b_state2
    output=self.mdl_o(state)
    return state,output #<NOTE:output> The output is of shape [_emp_reps,batch_size,num_output_features] (num_output_features=output_dimension)


class SsLayer(keras.layers.Layer):

    def __init__(self,ss_cell, return_outputs=True,instance_norm_flag=False,**kwargs):
        super().__init__(**kwargs)
        self.num_states = ss_cell.num_states
        self.output_dim = ss_cell.output_dim
        self.ss_cell = ss_cell
        self.state_initializer = tf.random_normal_initializer()
        self.return_outputs = return_outputs
        self.call_exec_time = None  # <TODO> remove this later
        # self.hidden_state=None  #<HERE>
        self.batch_size=None
        self.num_time_steps=None

        #regularization params of A and B
        self.alpha_A=0.1
        self.alpha_B=0.1

        #define an instance normalization layer
        self.instance_norm=keras.layers.Lambda(lambda x: (x,0*x,0*x))
        if instance_norm_flag:
            self.instance_norm=InstanceNormalization(name='inst-norm-layer',num_vars=self.output_dim) #the input sequence to the layer has the number of variables as the output sequence

    # def build(self,inputs_shape):
    #   pdb.set_trace()
    #   batch_size=inputs_shape[0] #should I use this or  tf.shape(inputs_shape)[0] <QUESTION?>
    #   self.hidden_state=self.get_init_state(batch_size=batch_size)

    def get_init_state(self, batch_size,dtype):
        temp = tf.zeros(shape=[batch_size, self.num_states], dtype=dtype)  # <HERE>
        # pdb.set_trace()
        return temp
    def build(self,input_shape):
        # The follwoing is done to ensure that matrix B in ss_cell is defined dynamically without raising bugs
        #as it is set to None at first
        input_shape=[input_shape[-1]]
        state_shape=[self.num_states]
        self.ss_cell.build([input_shape,state_shape])

    @tf.function(reduce_retracing=True)#,input_signature=[tf.TensorSpec(dtype='float32',shape=[None,None,None])])
    def call(self, input):
        input,mu,std=self.instance_norm(input) #perform instance normalization if the instance normalization flag is set to true
        # input has the shape of [batch, time, features]=[0,1,2]
        # convert the shape to [time,batch,features]=[1,0,2] (this will ease indexing the dataset)
        # pdb.set_trace()
        #pdb.set_trace()
        input = tf.transpose(input, perm=[1, 0, 2])
        self.num_time_steps = tf.shape(input)[0]  # 0 since the input has been restructured
        self.batch_size = tf.shape(input)[1]  # this is used to initialize the state (hidden-internal) state
        state = self.get_init_state(self.batch_size,input.dtype)  # initialize a state variable
        # pdb.set_trace()
        outputs_tensor = tf.TensorArray(dtype='float32', size=self.num_time_steps)
        # states_tensor=None#tf.TensorArray(dtype='float32',size=num_time_steps)

        #add regularization terms on A,B
        #self.add_loss(self.alpha_A*tf.reduce_sum(self.ss_cell.A)) #l1 regularization results in sparse matrices
        #self.add_loss(tf.linalg.norm(self.ss_cell.B,ord=1))


        for i in tf.range(self.num_time_steps):
            state_out=state #this is not ideal. I have to find an efficient method of returning the second to last hidden state
            state, output = self.ss_cell([input[i], state])  # SS_Cell expects two inputs [input,state]
            # pdb.set_trace()

            if self.return_outputs:
                #pdb.set_trace()
                outputs_tensor = outputs_tensor.write(i, tf.stack([output.parameters['loc'], output.parameters['scale_diag']],
                                                                  axis=0))  # for a Normal distribution, both the lco and scale parameters are of shape [batch_size,features]. Hence, stacking them shoudl result in [2*batch_size,features]
            # if self.return_states:                                                                    #<HERE> I left the states tensor for now
            #   if states_tensor is None:
            #     states_tensor=tf.TensorArray(dtype='float32',size=num_time_steps)
            #   states_tensor=states_tensor.write(i,self.hidden_state
        outputs_tensor=tf.transpose(outputs_tensor.stack(),perm=[1,2,0,3]) #shape =[2,batch_size,time_steps,output_dim]

        return outputs_tensor, state_out

    def __repr__(self):
        return '\n'.join([
                             'This is a customized recurrent neural layer that simulates a non-linear state space model. It can be used in a sequential model. To return the internal states, set the return_states flag to True. The same applies to the output.'])


class SsLayerSout(SsLayer):
#this is a class that inherits from SsLayer and overwrites the call method such that only the output sequence is returned and not the state
#(Why did I do it?) To be able to use mdl.fit for rapid testing; since mdl.fit works if the model generates one output. TODO: I just realized that this is not true and I could have modified the loss function instead :)
    def __init__(self,ss_cell,return_outputs=True,**kwargs):
        super().__init__(ss_cell=ss_cell,return_outputs=return_outputs,**kwargs)

    def call(self,input):
        outputs_tensor,_=super(SsLayerSout,self).call(input)
        return outputs_tensor


class SsRnnMdl(keras.Model):

    def __init__(self,ss_layer,**kwargs):  # num_states=50,output_dimension=10,return_states=False,return_outputs=True
        super().__init__(**kwargs)
        self.ss_layer =ss_layer

    def call(self, inputs):
        return self.ss_layer(inputs) #TODO: this is a temporary solution to return only one output (prediction). [New] I removed the temporary solution and I will
        #account for the second output in the loss function


class InstanceNormalization(keras.layers.Layer):

    def __init__(self,num_vars,**kwargs):
        super().__init__(**kwargs)
        self.time_axis = 1
        self.num_vars=num_vars
        self.initalizer=keras.initializers.GlorotNormal(seed=1)
        self.alpha = tf.Variable(initial_value=tf.ones(shape=[num_vars],dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(initial_value=tf.zeros(shape=[num_vars],dtype=tf.float32), trainable=True)

        self.eps=10**-3

    def call(self, x):
        mu = tf.reduce_mean(x, axis=self.time_axis)[:, tf.newaxis, :]
        std = tf.math.reduce_std(x, axis=self.time_axis)[:, tf.newaxis, :]
        return tf.matmul( (x-mu)/(std+self.eps) ,  tf.linalg.diag(self.alpha) ) + self.beta , mu, std

    def revert(self, x, mu, std):
        return std * (x - self.beta) / self.alpha + mu

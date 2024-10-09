import tensorflow as tf
from tensorflow.keras import layers,losses,metrics , optimizers,models,utils

General_Config = {
    'image_shape' : (512,512),
    'n_channel' : 3,
    'n_class' : 1,
}

Model_Config = {
    'n_level' : 3,
    'n_unit' : 32, ## start Units  
    'scaling_factor' : 2,
    'kernel_size' : 3,
    'n_block_layer' : 2, ## Number of layers in Each Block ..
    'activation' : 'relu',
    'strides':1,
    'dilation':2,
    'decoder_up':['upsample' , 'transpose'][0],
    'decoder_units' : 'same',
    'recurrent_time' : 2,
}

def ConvBlock(x ,n_unit = 64):
    x = layers.Conv2D(
                    filters = n_unit, 
                    kernel_size =  Model_Config['kernel_size'],
                    strides = Model_Config['strides'],
                    padding='same',
                    dilation_rate=Model_Config['dilation']
                    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def Simple_Block(x_input ,n_unit = 64):
    x_input = layers.Conv2D(n_unit,1,
                            strides = 1,
                            padding='same',
                            activation = 'relu',
                            dilation_rate=Model_Config['dilation'],
                           )(x_input)

    n_layer = Model_Config['n_block_layer']
    x_out = x_input 
    
    for j in range(n_layer):     
        x_out = ConvBlock(x_out,n_unit)

    return x_out

def Residual_Block( x_input ,n_unit = 64):
    x_input = layers.Conv2D(n_unit,1,
                            strides = 1,
                            padding='same',
                            activation = 'relu',
                            dilation_rate=Model_Config['dilation'],
                           )(x_input)

    n_layer = Model_Config['n_block_layer']
    x_out = x_input 
    
    for j in range(n_layer):     
        x_out = ConvBlock(x_out,n_unit)
    
    x_add = layers.Add()([x_out ,x_input])
    x_out = ConvBlock(x_add,n_unit)
    return x_out

def Recurrent_Block( x_input ,n_unit = 64):
    x_input = layers.Conv2D(n_unit,1,
                            strides = 1,
                            padding='same',
                            activation = 'relu',
                            dilation_rate=Model_Config['dilation'],
                           )(x_input)

    n_layer = Model_Config['n_block_layer']
    time = Model_Config['recurrent_time'] 
    x_out = x_input 

    for j in range(n_layer):      ## time is same as n_layer.
        x_res = ConvBlock(x_out,n_unit)

        for _ in range(time):            
            x_add = layers.Add()([x_out,x_res])
            x_res = ConvBlock(x_add,n_unit)
            
        x_out = x_res  ## save value to x to use in next iteration..

    return x_out

def R2_Block( x_input ,n_unit = 64):
    x_input = layers.Conv2D(n_unit,1,
                            strides = 1,
                            padding='same',
                            activation = 'relu',
                            dilation_rate=Model_Config['dilation'],
                           )(x_input)

    n_layer = Model_Config['n_block_layer']
    time = Model_Config['recurrent_time'] 
    x_out = x_input 
    for j in range(n_layer):      ## time is same as n_layer.
        x_res = ConvBlock(x_out,n_unit)
        
        for _ in range(time):            
            x_add = layers.Add()([x_out,x_res])
            x_res = ConvBlock(x_add,n_unit)
            
        x_out = x_res  ## save value to x to use in next iteration..

    x_add = layers.Add()([x_out ,x_input])
    x_out = ConvBlock(x_add,n_unit)
    return x_out

def Unit_Block(x_input ,n_unit = 64,num = 0):
    if(num==0):
        x_output = Simple_Block(x_input,n_unit) 
    elif(num==1):
        x_output = Residual_Block(x_input,n_unit) 
    elif(num==2):
        x_output = Recurrent_Block(x_input,n_unit) 
    else:
        x_output = R2_Block(x_input,n_unit) 
    return x_output;

def Decoder_Block1(x,y):
    scale = Model_Config['scaling_factor']

    if(Model_Config['decoder_up']=='transpose'):
        n_unit = x.shape[-1]
        x = layers.Conv2DTranspose(n_unit ,scale ,strides = scale ,activation = 'relu')(x)
    else:
        x = layers.UpSampling2D(scale)(x)
    
    x = layers.Concatenate()([x,y])
    return x

def Decoder_Block2(x,y_skip): ## With attenction
    scale = Model_Config['scaling_factor']
    
    n_unit = y_skip.shape[-1]
    
    y = layers.Conv2D(n_unit,1,strides = scale, padding='same', activation = 'relu')(y_skip)
    x = layers.Conv2D(n_unit,1,strides = 1, padding='same', activation = 'relu')(x)
    y = layers.Add()([x,y])
    y = layers.Activation('relu')(y)
    
    y = layers.Conv2D(1 ,1,padding='same',strides = 1)(y)

    if(Model_Config['decoder_up']=='transpose'):
        y = layers.Conv2DTranspose(1,scale ,scale ,activation = 'sigmoid')(y)
    else:
        y = layers.UpSampling2D(scale)(y)
        
    y = layers.Activation('tanh')(y)

    y = layers.Multiply()([y,y_skip])
    y = layers.Activation('relu')(y)
    return y
    
def Decoder_Block3(x,y_skip): ## With attenction
    scale = Model_Config['scaling_factor']
    
    n_unit = y_skip.shape[-1]
    
    y = layers.Conv2D(n_unit,1,strides = scale, padding='same')(y_skip)
    x = layers.Conv2D(n_unit,1,strides = 1, padding='same')(x)
    
    y = layers.Add()([x,y])
    y = layers.Activation('relu')(y)
    
    y = layers.Conv2D(1 ,1,padding='same',strides = 1)(y)

    if(Model_Config['decoder_up']=='transpose'):
        y = layers.Conv2DTranspose(1,scale ,scale ,activation = 'sigmoid')(y)
    else:
        y = layers.UpSampling2D(scale)(y)
        
    y = layers.Activation('sigmoid')(y)

    y = layers.Multiply()([y,y_skip])
    return y


def Encode(x,num):
    scale = Model_Config['scaling_factor']
    n_level = Model_Config['n_level']
    n_unit = Model_Config['n_unit']
    
    Encoder_List = []
    for i in range(n_level):
        x = Unit_Block(x,n_unit,num)

        Encoder_List.append(x)
        x = layers.MaxPool2D(scale,scale)(x)
        n_unit *= scale
    return x , Encoder_List

def Decode(x,Encoder_List , n_unit,num, has_attention= False):
    scale = Model_Config['scaling_factor']
    n_level = Model_Config['n_level']
    
    for i in range(n_level):
        y = Encoder_List.pop()
        if(has_attention):
            # x = Decoder_Block2(x,y)
            x = Decoder_Block3(x,y)
        else:
            x = Decoder_Block1(x,y)

        x = Unit_Block(x,n_unit,num)
        n_unit = n_unit//scale
    return x

Backbone_Call = {
    'vgg_16': tf.keras.applications.vgg16.VGG16,
    'vgg_19': tf.keras.applications.vgg19.VGG19,
    'efficientnet_b0': tf.keras.applications.efficientnet.EfficientNetB0,
}
Backbone_Preprocess = {
    'vgg_16':tf.keras.applications.vgg16.preprocess_input,
    'vgg_19':tf.keras.applications.vgg19.preprocess_input,
    'efficientnet_b0': tf.keras.applications.efficientnet.preprocess_input,
}
Backbone_Layer_Interval = {
    'vgg_16' :{
        0:['block1_conv1' , 'block1_conv2'],
        1:['block2_conv1' , 'block2_conv2'],
        2:['block3_conv1' , 'block3_conv3'],
        3:['block4_conv1' , 'block4_conv3'],
        4:['block5_conv1' , 'block5_conv3'],
    },
    'vgg_19' :{
        0:['block1_conv1' , 'block1_conv2'],
        1:['block2_conv1' , 'block2_conv2'],
        2:['block3_conv1' ,'block3_conv4'],
        3:['block4_conv1' , 'block4_conv4'],
        4:['block5_conv1' , 'block5_conv4'],
    },
    'efficientnet_b0' :{
        0:['block1_conv1' , 'block1_conv2'],
        1:['block2_conv1' , 'block2_conv2'],
        2:['block3_conv1' ,'block3_conv4'],
        3:['block4_conv1' , 'block4_conv4'],
        4:['block5_conv1' , 'block5_conv4'],
    },
}

# Model.get_layer()

def Unit_BackBone(x , Layer_Interval , back_bone):
    block_input = back_bone.get_layer(Layer_Interval[0]).input
    block_output = back_bone.get_layer(Layer_Interval[1]).output
    mini_model = models.Model(block_input ,block_output)
    x = mini_model(x)
    return x
    
def Encode2(x,image_shape ,backbone_name = 'vgg_16'):

    back_bone = Backbone_Call[backbone_name](include_top=False,input_shape = image_shape ,weights='imagenet')
    # x = layers.Lambda( lambda x:Backbone_Preprocess[backbone_name](x) , output_shape = image_shape ,mask= False,)(x)
    x = Backbone_Preprocess[backbone_name](x)
    
    n_level = Model_Config['n_level']
    scale = Model_Config['scaling_factor']
    
    Encoder_List = []
    for i in range(n_level):
        x = Unit_BackBone(x ,Backbone_Layer_Interval[backbone_name][i] ,back_bone )
        Encoder_List.append(x)
        x = layers.MaxPool2D(scale,scale)(x)
    
    return x , Encoder_List


def BuildUnet(backbone=None , num = 0,has_attention = False):
    image_shape = General_Config['image_shape']+(General_Config['n_channel'],)
    n_class = General_Config['n_class']
    
    fn_input = layers.Input(shape = image_shape)
    ## Encoding ..............................
    # x_enc , Encoder_List  = Encode(fn_input , num )
    if(backbone):
        x_enc , Encoder_List = Encode2(fn_input,image_shape,backbone)
    else :
        x_enc, Encoder_List  = Encode(fn_input , num)

    scale = Model_Config['scaling_factor']
    n_unit = x_enc.shape[-1] 
    
    ## Base Part ............................
    x_base = Unit_Block(x_enc ,n_unit *scale , num)
    ## Decoder Part..........................
    x_dec = Decode(x_base,Encoder_List ,n_unit ,num, has_attention)
    ## Ending Part ..........................
    
    fn_output = layers.Conv2D(1 ,Model_Config['kernel_size'] ,padding = 'same')(x_dec)
    if(n_class==1):
        fn_output = layers.Activation('sigmoid')(fn_output)
    else:
        fn_output = layers.Activation('softmax')(fn_output)
        
    ### Build Model ==================================================|
    model_name = "Unet_Model"
    if (backbone):
        model_name = backbone + '_'+model_name
    Model = models.Model(fn_input ,fn_output,name = model_name)
    return Model

# Model = BuildUnet(backbone= 'vgg_16',num=0,has_attention= True)

# Model.summary()

# tf.keras.config.enable_unsafe_deserialization()
# models.clone_model(Model)

# image_shape = General_Config['image_shape']+(General_Config['n_channel'],)
# Model.build(image_shape)
# # Model(tf.zeros((1,)+image_shape,dtype='float32'))

# utils.plot_model(Model,show_shapes = True,show_layer_names=True)

# image_shape = General_Config['image_shape'] + (General_Config['n_channel'],)
# pre_model = Backbone_Call['efficientnet_b0'](include_top=False,input_shape = image_shape ,weights='imagenet')

# utils.plot_model(pre_model,show_shapes = True,show_layer_names=True)

# pre_process_model = Backbone_Preprocess['vgg_16']


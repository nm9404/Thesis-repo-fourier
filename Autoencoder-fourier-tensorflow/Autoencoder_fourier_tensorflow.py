import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from PIL import Image
import numpy as np
from FourierUtils import FourierUtils

class FourierAutoencoder:
    def __init__(self, image, w_init, sigma, num_iters=4000):
        self.image=image
        self.sigma=sigma
        self.w_init=w_init
        self.num_iters=num_iters

    def centralFilterNeuron(self, freqSpec, muVar, sigmaVar, w1, w0):    
        lowerLimit=tf.constant([0., 0.], tf.float32)
        tensorShape=freqSpec.get_shape().as_list()
        higherLimit=tf.constant([tensorShape[1], tensorShape[2]], tf.float32)
    
        magnitudeTensor=freqSpec[0]
        phaseTensor=freqSpec[1]
    
        translationTransformation=tf.constant([1./2.*float(tensorShape[1]), 1./2.*float(tensorShape[2])])
        multiplicationTransformation=tf.constant([1., -1.])
        imShape=tf.constant([float(tensorShape[1]), float(tensorShape[2])])
    
        muVar=tf.multiply((tf.scalar_mul(1./2., imShape))/np.pi, muVar)
    
        muVar=tf.multiply(muVar, multiplicationTransformation)
        muVar=tf.add(muVar, translationTransformation)
    
        muX=muVar[0]
        muY=muVar[1]
    
        muXMatrix=tf.scalar_mul(-1.,tf.scalar_mul(muX,tf.ones([tensorShape[1], tensorShape[2]])))
        muYMatrix=tf.scalar_mul(-1.,tf.scalar_mul(muY,tf.ones([tensorShape[1], tensorShape[2]])))
    
        x = tf.range(0, imShape[1], 1)
        y = tf.range(0, imShape[0], 1)
        mesh = tf.ones([tensorShape[1], tensorShape[2]])
    
        meshX, meshY = tf.meshgrid(x,y)
    
        meshX=tf.add(meshX,mesh)
        meshY=tf.add(meshY,mesh)
    
        meshXt=tf.add(meshX,muXMatrix)
        meshYt=tf.add(meshY,muYMatrix)
    
        meshXSquared=tf.square(meshXt)
        meshYSquared=tf.square(meshYt)
        addition=tf.add(meshXSquared,meshYSquared)
    
        sigm=tf.reduce_mean(sigmaVar)
    
        exponent=tf.scalar_mul(1/sigm,tf.scalar_mul(-1., addition))
        gaussianSample=tf.exp(exponent);
        gaussianSampleCorrec=tf.add(tf.scalar_mul(1e-10,mesh),gaussianSample) #Hacerle clip a la exponencial para que la derivada no muera

        return tf.stack([gaussianSampleCorrec, tf.ones([tensorShape[1], tensorShape[2]])])

    def filterImage(self, freqSpec, filterStack):
        return tf.stack([tf.multiply(freqSpec[0], filterStack[0]) ,tf.multiply(freqSpec[1], filterStack[1])])

    def neuronFilter(self, freqSpec, muVar, sigmaVar, w1, w0):
        return self.filterImage(freqSpec, centralFilterNeuron(freqSpec, muVar, sigmaVar, w1, w0))

    def cartToPolar(self, complexTuple):
        realPart=complexTuple[0]
        complexPart=complexTuple[1]
        magnitude=tf.sqrt(tf.add(tf.multiply(realPart, realPart), tf.multiply(complexPart, complexPart)))
        phase=tf.atan2(complexPart, realPart)
        return tf.stack([magnitude, phase])

    def polarToCart(self, complexTuple):
        magnitude=complexTuple[0]
        phase=complexTuple[1]
        realPart=tf.multiply(magnitude, tf.cos(phase))
        complexPart=tf.multiply(magnitude, tf.sin(phase))
        return tf.stack([realPart, complexPart])

    def log10(self, x):
        numerator=tf.log(x)
        denominator=tf.log(10.)
        return tf.scalar_mul(1/denominator, numerator)

    def sumOperator(self, filterTensor, weights):
    
        mesh=tf.ones([tf.shape(filterTensor)[2], tf.shape(filterTensor)[3]], dtype=tf.float32)
        meshTuple=tf.stack([mesh,mesh])
        weightsMatrices=tf.map_fn(lambda x: tf.scalar_mul(x,meshTuple), weights)
    
        filterTensorWeighted=tf.multiply(weightsMatrices,filterTensor)
    
        cartParts=tf.map_fn(lambda x: self.polarToCart(x), filterTensorWeighted)
        originalParts=tf.map_fn(lambda x: self.polarToCart(x), filterTensor)
    

        filteredParts=tf.reduce_sum(cartParts, 0)
        filteredPartsNoWeight=self.cartToPolar(tf.reduce_sum(originalParts,0))
        filteredPartsPolar=self.cartToPolar(filteredParts)
    
        filteredPartsPolarNoPha=[filteredPartsPolar[0],filteredPartsNoWeight[1]]
    
        return filteredPartsPolarNoPha

    def fftunshift(self, spectrum):
        x = tf.range(0, tf.shape(spectrum)[1], 1)
        y = tf.range(0, tf.shape(spectrum)[0], 1)
    
        meshX, meshY = tf.meshgrid(x,y)
    
        meshX=tf.cast(meshX, dtype=tf.float32)
        meshY=tf.cast(meshY, dtype=tf.float32)
    
        mesh=tf.ones([tf.shape(spectrum)[0], tf.shape(spectrum)[1]], dtype=tf.float32)
    
        meshX = meshX+mesh
        meshY = meshY+mesh
    
        base=tf.scalar_mul(-1., mesh)
        Bx=tf.pow(base, tf.add(meshX,meshY))
    
        return tf.multiply(spectrum, Bx)

    def correctBank(self, filterBank, weights):
        summedBankPolar = self.sumOperator(filterBank, weights)
        baseMatrix = tf.ones([tf.shape(summedBankPolar)[1], tf.shape(summedBankPolar)[2]], dtype=tf.float32)
        compMatrix = tf.greater(summedBankPolar,[baseMatrix,baseMatrix])
        compMatrix = tf.cast(compMatrix, dtype=tf.float32)
        subA = tf.subtract(summedBankPolar, baseMatrix)
        multMatrix = tf.multiply(compMatrix, subA)
        return tf.add(baseMatrix, multMatrix)

    def correctBankMean(self, filterBank):
        baseMatrix = tf.ones([tf.shape(filterBank)[2], tf.shape(filterBank)[3]])
        bankNew = tf.map_fn(lambda x: tf.greater_equal(x, [tf.scalar_mul(0.01, baseMatrix), tf.scalar_mul(0.01, baseMatrix)]) ,filterBank, dtype=tf.bool)
        bankNew = tf.cast(bankNew, dtype=tf.float32)
        bankSummed = tf.reduce_sum(bankNew, 0)
    
        complement = tf.less_equal(bankSummed, [tf.scalar_mul(0., baseMatrix), tf.scalar_mul(0., baseMatrix)])
        complement = tf.cast(complement, dtype=tf.float32)

        correctedBank = tf.add(bankSummed, complement)
        return tf.stack([correctedBank[0], baseMatrix])

    def PSNR(self, imageA,imageB,maxVal):
        loss=tf.reduce_mean(tf.square(imageA-imageB))
        return tf.scalar_mul(10, log10((maxVal**2)/loss))

    def SSIM(self, imageA,imageB,maxVal):
        varA=tfp.stats.variance(imageA, sample_axis=[0,1])
        varB=tfp.stats.variance(imageB, sample_axis=[0,1])
        covAB=tfp.stats.covariance(imageA,imageB,sample_axis=[0,1],event_axis=None)
        meanA=tf.reduce_mean(imageA)
        meanB=tf.reduce_mean(imageB)
        k1=0.01
        k2=0.03
        c1=(k1*maxVal)**2
        c2=(k2*maxVal)**2
        num=(2*meanA*meanB+c1)*(2*covAB+c2)
        den=(meanA**2+meanB**2+c1)*(varA+varB+c2)
        return num/den


    def L1Norm(self, tensor,lamb):
        norm=tf.norm(tensor,ord=1)
        return tf.scalar_mul(lamb,norm)

    def adaptLamb(self, lamb,loss,maxRange):
        return lamb*loss/maxRange

    def estimate_W(self):
        fourier_utils=FourierUtils(self.image)
        transform=fourier_utils.transform()
        magnitudeTensor=tf.convert_to_tensor(transform[0], np.float32)
        phaseTensor=tf.convert_to_tensor(transform[1], np.float32)
        imageTensor=tf.convert_to_tensor(self.image, np.float32)
        data = (transform[0], transform[1])

        tf.disable_v2_behavior()
        g = tf.Graph()
        Ob = []
        errorLoss=[]
        
        with g.as_default():
            X = tf.placeholder(tf.float32, shape=[2,np.shape(self.image)[0],np.shape(self.image)[1]])
            sigmaVar = tf.constant(self.sigma, tf.float32)
            w0 = tf.constant([0], tf.float32)
            w1 = tf.constant([1], tf.float32)

            with tf.name_scope('operations') as scope:
                #Configurar pesos
                weights=tf.Variable([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.], tf.float32, name='W',constraint=lambda t: tf.clip_by_value(t, 1e-10, 1))
  
                muVars = tf.Variable(self.w_init, tf.float32, name='O')
        
                muVarCenter=tf.constant([[0.,0.],[0.,0.]])
                elems_center=muVarCenter
                sigmaCenter=tf.constant([26500./700., 26500./700.])
                centralFilter=tf.map_fn(lambda x: self.centralFilterNeuron(X,x,sigmaCenter,w0,w1),elems_center)
                elems=(muVars)
                filterBank = tf.map_fn(lambda x: self.centralFilterNeuron(X, x, sigmaVar, w0, w1), elems)
        
                filterBankWithCenter = tf.concat([filterBank,centralFilter],0)
        
                filterBankEval = self.sumOperator(filterBankWithCenter, weights)
                correctionMatrix = self.correctBank(filterBankWithCenter, weights)

                evalFilters= tf.map_fn(lambda x: self.filterImage(X,x), filterBankWithCenter)
                sumFilter = self.sumOperator(evalFilters, weights)
        
                sumFilterP  = tf.divide(sumFilter, correctionMatrix)
                sumFilter = self.polarToCart(sumFilterP)

                complexTuple=tf.complex(sumFilter[0], sumFilter[1])        
                fourierInvertedN=tf.ifft2d(complexTuple)        
                fourierInverted=(tf.abs(fourierInvertedN))
        
            with tf.name_scope('loss') as scope:        
                imageA=fourierInverted
                imageB=tf.convert_to_tensor(self.image.astype('float32'))
                PSNRLoss=tf.scalar_mul(-1.,self.SSIM(imageA,imageB,255)+1)
                mixedLoss=PSNRLoss + self.L1Norm(weights,self.adaptLamb(1/200.,PSNRLoss,-40.))        
        
            with tf.name_scope('training') as scope:
                lr = 0.5
                optimizer = tf.train.GradientDescentOptimizer(lr)
                train = optimizer.minimize(PSNRLoss)
        
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
            
                    for step in range(self.num_iters):
                        _, finalLoss, finalReconstruction = sess.run([train, PSNRLoss, fourierInverted], feed_dict={X:data})
                    final_w=sess.run([muVars])
        return final_w

               





import numpy
import tensorflow as tf
from copy import deepcopy


class ParamsRecurrentMultiple:
    def __init__(self):
        self.nomeTipoRede: str = ""
        self.shapeEntrada: tuple = ()
        self.learningRate: float = 1e-3
        self.qtdeNeuroniosEntrada: int = 0
        self.arrQtdeNeuroniosOculta: list[int] = []
        self.arrNomesAtivacaoOculta: list[str] = []
        self.arrQtdeNeuroniosSaida: list[int] = []
        self.arrNomesAtivacaoSaida: list[str] = []
        self.arrNameLosses: list[str] = []

        self.nEpocas: int = 5000
        self.qtdeDadosValidar: int = 20
        self.batchSize: int = 25

        self.datasetEntrada: list = []
        self.datasetRotulo: list = []
        self.datasetPrever: list = []


class NetworkRecurrentMultiple:
    def train(self, params: ParamsRecurrentMultiple, model=None, isTrain=True):

        if model is None:
            model = self.buildModelNetworkMultipleOutput(shapeEntrada=params.shapeEntrada,
                                                         arrnNeuroniosOculta=params.arrQtdeNeuroniosOculta,
                                                         arrnNeuroniosSaida=params.arrQtdeNeuroniosSaida,
                                                         arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida)

        if isTrain:
            model = self.treinarModelNetwork(
                model=model, datasetEntradaMain=params.datasetEntrada,
                datasetRotuloMain=params.datasetRotulo,
                batchSize=params.batchSize, qtdeDadosValidar=params.qtdeDadosValidar,
                arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                nEpocas=params.nEpocas)

        arrPredicts = []
        if len(params.datasetPrever) >= 1:
            arrPredicts = self.previsaoModelNetwork(model=model,
                                                    datasetEntrada=params.datasetEntrada,
                                                    datasetPrever=params.datasetPrever,
                                                    isMultipleOutput=len(params.arrNomesAtivacaoSaida) >= 2)

        msgPrev = "Previsões: " + str(arrPredicts)
        return model, arrPredicts, msgPrev

    def trainOnBatchOneTeamBatch(self, params: ParamsRecurrentMultiple, model=None, isTrain=True):
        entradas = deepcopy(params.datasetEntrada)
        rotulos = deepcopy(params.datasetRotulo)
        if model is None:
            model = self.buildModelNetworkMultipleOutput(shapeEntrada=params.shapeEntrada,
                                                         arrnNeuroniosOculta=params.arrQtdeNeuroniosOculta,
                                                         arrnNeuroniosSaida=params.arrQtdeNeuroniosSaida,
                                                         arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida)

        if isTrain:
            arrEntPA = [entradas[-2][0], entradas[-1][0]]
            entradas = entradas[:len(entradas) - 2]

            arrRotPA = [rotulos[-2][0][0], rotulos[-1][0][0]]
            rotulos = rotulos[:len(rotulos) - 2]

            model = self.treinarModelNetworkOnBatch(
                model=model, datasetEntradaMain=entradas,
                datasetRotuloMain=rotulos,
                qtdeDadosValidar=params.qtdeDadosValidar,
                arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                nEpocas=params.nEpocas)

            arrPredicts, msg = self.previsaoModelNetwork(model=model, datasetEntrada=entradas,
                                                         datasetPrever=[arrEntPA],
                                                         isMultipleOutput=len(params.arrNomesAtivacaoSaida) >= 2)
            arrpreds = arrPredicts[0]
            qtdeAcertos = 0
            for idxRot, rot in enumerate(arrRotPA):
                maxArgRot = numpy.argmax(rot)
                maxArgPrv = numpy.argmax(arrpreds[idxRot])
                if maxArgPrv == maxArgRot:
                    qtdeAcertos += 1

            if qtdeAcertos >= 2:
                arrPredicts = []
                msg = "Nenhuma previsão"
                if len(params.datasetPrever) >= 1:
                    for entPA in arrEntPA:
                        entradas.append([entPA])

                    for rotPA in arrRotPA:
                        rotulos.append([[rotPA]])

                    model = self.treinarModelNetworkOnBatch(
                        model=model, datasetEntradaMain=entradas,
                        datasetRotuloMain=rotulos,
                        qtdeDadosValidar=params.qtdeDadosValidar,
                        arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                        nEpocas=params.nEpocas)

                    arrPredicts, msg = self.previsaoModelNetwork(model=model, datasetEntrada=entradas,
                                                                 datasetPrever=params.datasetPrever,
                                                                 isMultipleOutput=len(
                                                                     params.arrNomesAtivacaoSaida) >= 2)

                msgPrev = "Previsões: \n" + str(msg)
                return model, arrPredicts, msgPrev
            else:
                print(arrRotPA)
                print(arrpreds.tolist())
                return False

    def trainOnBatchMoreTeamsBatch(self, params: ParamsRecurrentMultiple, model=None, isTrain=True):
        entradas = deepcopy(params.datasetEntrada)
        rotulos = deepcopy(params.datasetRotulo)
        if model is None:
            model = self.buildModelNetworkMultipleOutput(shapeEntrada=params.shapeEntrada,
                                                         arrnNeuroniosOculta=params.arrQtdeNeuroniosOculta,
                                                         arrnNeuroniosSaida=params.arrQtdeNeuroniosSaida,
                                                         arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida)

        if isTrain:
            arrEntPA = entradas[0][-3:]
            entradas[0] = entradas[0][:len(entradas[0]) - 3]

            arrRotPA = rotulos[0][0][-3:]
            rotulos[0][0] = rotulos[0][0][:len(rotulos[0][0]) - 3]

            model = self.treinarModelNetworkOnBatch(
                model=model, datasetEntradaMain=entradas,
                datasetRotuloMain=rotulos,
                qtdeDadosValidar=params.qtdeDadosValidar,
                arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                nEpocas=params.nEpocas)

            arrPredicts, msg = self.previsaoModelNetwork(model=model, datasetEntrada=entradas,
                                                         datasetPrever=[arrEntPA],
                                                         isMultipleOutput=len(params.arrNomesAtivacaoSaida) >= 2)
            arrpreds = arrPredicts[0]
            qtdeAcertos = 0
            for idxRot, rot in enumerate(arrRotPA):
                maxArgRot = numpy.argmax(rot)
                maxArgPrv = numpy.argmax(arrpreds[idxRot])
                if maxArgPrv == maxArgRot:
                    qtdeAcertos += 1

            if qtdeAcertos >= 2:
                arrPredicts = []
                msg = "Nenhuma previsão"
                if len(params.datasetPrever) >= 1:
                    entradas[0] = numpy.concatenate((entradas[0], arrEntPA)).tolist()
                    rotulos[0][0] = numpy.concatenate((rotulos[0][0], arrRotPA)).tolist()

                    model = self.treinarModelNetworkOnBatch(
                        model=model, datasetEntradaMain=entradas,
                        datasetRotuloMain=rotulos,
                        qtdeDadosValidar=params.qtdeDadosValidar,
                        arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                        nEpocas=params.nEpocas)

                    arrPredicts, msg = self.previsaoModelNetwork(model=model, datasetEntrada=entradas,
                                                                 datasetPrever=params.datasetPrever,
                                                                 isMultipleOutput=len(
                                                                     params.arrNomesAtivacaoSaida) >= 2)

                msgPrev = "Previsões: \n" + str(msg)
                return model, arrPredicts, msgPrev
            else:
                print(arrRotPA)
                print(arrpreds.tolist())
                return False

    def trainOnBatchBKP(self, params: ParamsRecurrentMultiple, model=None, isTrain=True):

        if model is None:
            model = self.buildModelNetworkMultipleOutput(shapeEntrada=params.shapeEntrada,
                                                         arrnNeuroniosOculta=params.arrQtdeNeuroniosOculta,
                                                         arrnNeuroniosSaida=params.arrQtdeNeuroniosSaida,
                                                         arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida)

        if isTrain:
            model = self.treinarModelNetworkOnBatch(
                model=model, datasetEntradaMain=params.datasetEntrada,
                datasetRotuloMain=params.datasetRotulo,
                qtdeDadosValidar=params.qtdeDadosValidar,
                arrFuncAtivacaoSaida=params.arrNomesAtivacaoSaida, learning_rate=params.learningRate,
                nEpocas=params.nEpocas)

        arrPredicts = []
        msg = "Nenhuma previsão"
        if len(params.datasetPrever) >= 1:
            arrPredicts, msg = self.previsaoModelNetwork(model=model, datasetEntrada=params.datasetEntrada,
                                                         datasetPrever=params.datasetPrever,
                                                         isMultipleOutput=len(params.arrNomesAtivacaoSaida) >= 2)

        msgPrev = "Previsões: \n" + str(msg)
        return model, arrPredicts, msgPrev

    @staticmethod
    def previsaoModelNetwork(model: tf.keras.Model, datasetEntrada: list, datasetPrever: list, isMultipleOutput: bool):
        msg = ""
        arrPredicts = []
        for idxPrev, prev in enumerate(datasetPrever):
            arrAllPredicts: list = model.predict(x=numpy.array(prev))

            if not isMultipleOutput:
                arrPredicts.append(arrAllPredicts)
                msg += "not" + " - " + str(arrAllPredicts) + "\n"
            else:
                maxA = 0
                maxB = 0
                for idxAllpred in range(len(arrAllPredicts)):
                    arrPred = []
                    for idxPred in range(len(arrAllPredicts[idxAllpred])):
                        arrPred.append(list(arrAllPredicts[idxAllpred][idxPred]))
                    msg += str(idxAllpred) + " - " + str(arrPred) + "\n"
                    arrPredicts.append(arrPred)
                maxA = numpy.array(arrAllPredicts[0][0]).argmax()
                maxB = numpy.array(arrAllPredicts[1][0]).argmax()
                saldoGols = maxA - maxB
                msg += "Saldo de gols de" + ": " + str(saldoGols) + "\n"

            msg += "-----------\n"
        return arrPredicts, msg

    def treinarModelNetwork(self, model: tf.keras.Model, datasetEntradaMain: list, datasetRotuloMain: list,
                            batchSize: int, qtdeDadosValidar: int, arrFuncAtivacaoSaida: list[str],
                            learning_rate: float, nEpocas: int):
        """
        :param model:
        :param datasetEntradaMain:
        :param datasetRotuloMain:
        :param batchSize:
        :param qtdeDadosValidar:
        :param arrFuncAtivacaoSaida:
        :param learning_rate:
        :param nEpocas:
        :return: retorna o model treinado, um array de todas as saidas e um array com os acertos das validacoes:
        """

        datasetRotulo = deepcopy(datasetRotuloMain)
        datasetEntrada = deepcopy(datasetEntradaMain)
        metrics = self.obterMetrics(arrFunctionsAtivacao=arrFuncAtivacaoSaida)
        functionLoss = self.obterFunctionsLoss(arrFunctionsAtivacao=arrFuncAtivacaoSaida)
        datasetEntradaTreinar = datasetEntrada
        validationData = None

        if qtdeDadosValidar >= 1:
            datasetEntradaTreinar, datasetEntradaValidar = (
                datasetEntrada[:-qtdeDadosValidar], datasetEntrada[-qtdeDadosValidar:])
            datasetRotuloTreinar, datasetRotuloValidar = (
                self.obterRotulos(arrRotulos=datasetRotulo, qtdeDadosValidar=qtdeDadosValidar))

            validationData = (numpy.array(datasetEntradaValidar), datasetRotuloValidar)
        else:
            datasetRotuloTreinar = self.obterRotulos(arrRotulos=datasetRotulo, qtdeDadosValidar=qtdeDadosValidar)[0]

        '''funcStoppingA = keras.callbacks.EarlyStopping(monitor="accuracy", patience=2000, mode="max",
                                                      baseline=0.99, min_delta=0)'''
        funcStoppingA = StopTrainingAtAccuracy()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics,
                      loss=functionLoss)
        model.fit(x=numpy.array(datasetEntradaTreinar), y=datasetRotuloTreinar,
                  validation_data=validationData,
                  callbacks=[funcStoppingA], epochs=nEpocas, batch_size=batchSize)

        return model

    def treinarModelNetworkOnBatch(self, model: tf.keras.Model, datasetEntradaMain: list, datasetRotuloMain: list,
                                   qtdeDadosValidar: int, arrFuncAtivacaoSaida: list[str],
                                   learning_rate: float, nEpocas: int):
        """
        :param model:
        :param datasetEntradaMain:
        :param datasetRotuloMain:
        :param qtdeDadosValidar:
        :param arrFuncAtivacaoSaida:
        :param learning_rate:
        :param nEpocas:
        :return: retorna o model treinado, um array de todas as saidas e um array com os acertos das validacoes:
        """
        datasetRotuloBatch = deepcopy(datasetRotuloMain)
        datasetEntradaBatch = deepcopy(datasetEntradaMain)
        metrics = self.obterMetrics(arrFunctionsAtivacao=arrFuncAtivacaoSaida)
        functionLoss = self.obterFunctionsLoss(arrFunctionsAtivacao=arrFuncAtivacaoSaida)
        arrRotulosNorms = []

        for rot in datasetRotuloBatch:
            datasetRotuloTreinar = self.obterRotulos(arrRotulos=rot,
                                                     qtdeDadosValidar=qtdeDadosValidar)[0]
            arrRotulosNorms.append(datasetRotuloTreinar)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=metrics,
                      loss=functionLoss)
        nCamadasOutput = len(datasetRotuloBatch[0])
        for nEpoca in range(nEpocas):
            sumAccuracy = 0
            sumLoss = 0
            lenBatch = len(datasetEntradaMain)
            msgEpoca = "--------------------------------------------------\n"
            msgEpoca += "nEpoca {} / {}, nBatchs: {}\n".format(nEpoca, nEpocas, lenBatch)
            for idxBatch in range(len(datasetEntradaMain)):
                datasetEntradaTreinar = datasetEntradaBatch[idxBatch]
                datasetRotuloTreinar = arrRotulosNorms[idxBatch]
                sumLossByLayer = 0
                sumAccuracyByLayer = 0

                retTrain = model.train_on_batch(x=numpy.array(datasetEntradaTreinar), y=datasetRotuloTreinar,
                                                return_dict=True)

                if nCamadasOutput >= 2:
                    for idxRotulo in range(nCamadasOutput):
                        nameKeyLoss = "output" + str(idxRotulo) + "_loss"
                        nameKeyAccuracy = "output" + str(idxRotulo) + "_accuracy"

                        sumLossByLayer += retTrain[nameKeyLoss]
                        sumAccuracyByLayer += retTrain[nameKeyAccuracy]

                else:
                    nameKeyLoss = "loss"
                    nameKeyAccuracy = "accuracy"

                    sumLossByLayer += retTrain[nameKeyLoss]
                    sumAccuracyByLayer += retTrain[nameKeyAccuracy]

                sumLoss += sumLossByLayer / nCamadasOutput
                sumAccuracy += sumAccuracyByLayer / nCamadasOutput

            mediaLoss = sumLoss / lenBatch
            mediaAccuracy = sumAccuracy / lenBatch
            msgEpoca += "Loss rede: {}, Accuracy rede: {}".format(mediaLoss, mediaAccuracy)

            print(msgEpoca)
            if mediaAccuracy >= 1 and mediaLoss <= 0.01:
                print("Atungiu accuracy máxima.")
                break

        return model

    @staticmethod
    def buildModelNetworkMultipleOutput(shapeEntrada: tuple, arrnNeuroniosOculta: list[int],
                                        arrnNeuroniosSaida: list[int], arrFuncAtivacaoSaida: list[str]):
        lInput = tf.keras.layers.Input(shape=shapeEntrada)
        lHidden = None
        nCamadas = len(arrnNeuroniosOculta)
        for idxCamada in range(nCamadas):
            nNeuronios = arrnNeuroniosOculta[idxCamada]
            isReturnSequancia = idxCamada < (nCamadas - 1)
            if idxCamada == 0:
                lHidden = tf.keras.layers.LSTM(units=nNeuronios, activation="tanh",
                                               return_sequences=isReturnSequancia,
                                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                               recurrent_initializer=tf.keras.initializers.GlorotNormal())(lInput)
            else:
                lHidden = tf.keras.layers.LSTM(units=nNeuronios, activation="tanh",
                                               return_sequences=isReturnSequancia,
                                               kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                               recurrent_initializer=tf.keras.initializers.GlorotNormal())(lHidden)

        outputs = []
        for idxLSaida in range(len(arrFuncAtivacaoSaida)):
            funcAtivacaoSaida = arrFuncAtivacaoSaida[idxLSaida]
            nNeurosSaida = arrnNeuroniosSaida[idxLSaida]
            if idxLSaida == 0:
                outputs.append(tf.keras.layers.Dense(units=nNeurosSaida, activation=funcAtivacaoSaida,
                                                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                     name=("output" + str(idxLSaida)))(lHidden))
            else:
                outputs.append(tf.keras.layers.Dense(units=nNeurosSaida, activation=funcAtivacaoSaida,
                                                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                     name=("output" + str(idxLSaida)))(lHidden))

        model = tf.keras.Model(inputs=lInput, outputs=outputs)
        return model

    @staticmethod
    def obterFunctionsLoss(arrFunctionsAtivacao: list[str]):
        dictLosses = {}
        for idxLo in range(len(arrFunctionsAtivacao)):
            if arrFunctionsAtivacao[idxLo] == "softmax":
                functionLoss = "categorical_crossentropy"
            elif arrFunctionsAtivacao[idxLo] == "sigmoid":
                functionLoss = "binary_crossentropy"
            else:
                functionLoss = "mse"
            dictLosses["output"+str(idxLo)] = functionLoss
        return dictLosses

    @staticmethod
    def obterMetrics(arrFunctionsAtivacao: list[str]):
        dictMetrics = {}
        for idxLo in range(len(arrFunctionsAtivacao)):
            dictMetrics["output" + str(idxLo)] = "accuracy"
        return dictMetrics

    @staticmethod
    def obterRotulos(arrRotulos: list, qtdeDadosValidar: int):
        dictRotulosTrain = {}
        dictRotulosValid = {}
        if qtdeDadosValidar >= 1:
            for idxLo in range(len(arrRotulos)):
                dictRotulosTrain["output" + str(idxLo)] = numpy.array(arrRotulos[idxLo][:-qtdeDadosValidar])
                dictRotulosValid["output" + str(idxLo)] = numpy.array(arrRotulos[idxLo][-qtdeDadosValidar:])
        else:
            for idxLo in range(len(arrRotulos)):
                dictRotulosTrain["output" + str(idxLo)] = numpy.array(arrRotulos[idxLo])
                dictRotulosValid["output" + str(idxLo)] = numpy.array(arrRotulos[idxLo])

        return dictRotulosTrain, dictRotulosValid


class StopTrainingAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy=1.0, target_loss=0.05):
        super(StopTrainingAtAccuracy, self).__init__()
        self.target_accuracy = target_accuracy
        self.target_loss = target_loss
        self.target_val_accuracy = 0.8

    def on_epoch_end(self, epoch, logs=None):
        current_accuracy = logs.get('accuracy')
        current_loss = logs.get('loss')
        current_val_accuracy_O1 = logs.get('val_output1_accuracy')
        current_val_accuracy = logs.get('val_accuracy')
        if (current_accuracy is not None and current_loss is not None
                and current_accuracy >= self.target_accuracy and current_loss <= self.target_loss):
            print(f"\nAtingiu a precisão desejada de {self.target_accuracy}. Parando o treinamento.")
            self.model.stop_training = True
        elif current_val_accuracy_O1 is not None and current_val_accuracy_O1 >= 0.8:
            print(f"\nAtingiu a precisão desejada de {self.target_accuracy}. Parando o treinamento usando val_acuracy.")
            self.model.stop_training = True
        elif (current_val_accuracy is not None and current_val_accuracy >= 0.9 and
              current_accuracy is not None and current_accuracy >= 0.80):
            print(f"\nAtingiu a precisão desejada de {self.target_accuracy}. Parando o treinamento usando val_acuracy.")
            self.model.stop_training = True

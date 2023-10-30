import numpy
import random
from copy import deepcopy

from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.losses import MSE

"""from keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam"""


from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy, BoltzmannQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory, Memory
from rl.core import Processor
from rl.callbacks import FileLogger, Callback

from gym import Env
from gym.spaces import Discrete, Box, Tuple

from api.regras.datasetPartidasRegras import DatasetPartidasRegras
from api.regras.iaUteisRegras import IAUteisRegras

class myCallBackStop(Callback):
    def on_episode_end(self, episode, logs={}):
        if logs["episode_reward"] == logs["nb_episode_steps"]:
            print(logs)

class FootEnv(Env):

    def __init__(self, datasetEntrada, datasetRotulo):
        # Numero de resultados possiveis ex: acertou e errou
        self.datasetEntrada = datasetEntrada
        self.datasetRotulo = datasetRotulo
        self.action_space = Discrete(len(datasetRotulo[0]))
        self.observation_space = Box(low=numpy.array([0., 0., 0.]), high=numpy.array([1., 1., 1.]), dtype=numpy.float32)

        # Um start para o state
        self.indexDado = 0
        self.state = numpy.array(datasetEntrada[self.indexDado])
        self.target = numpy.array(datasetRotulo[self.indexDado])
        self.n_amostras = len(datasetEntrada)

    def step(self, action):
        reward = numpy.array(self.datasetRotulo[self.indexDado])[action]
        self.indexDado += 1
        # Fim do processo
        done = self.indexDado >= len(self.datasetRotulo)

        if not done:
            self.state = self.datasetEntrada[self.indexDado]

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self, *, seed=None, options=None):
        self.indexDado = 0
        self.state = numpy.array(self.datasetEntrada[self.indexDado])
        self.target = numpy.array(self.datasetRotulo[self.indexDado])
        return self.state


class CustomProcessor(Processor):
    def process_state_batch(self, batch):
        newBatch = numpy.squeeze(batch, axis=1)
        return newBatch


class ViviFoot:
    def __init__(self, id_team_home: int, id_team_away: int = None, isAmbas: bool = True,
                 isAgruparTeams: bool = True, idTypeReturn: int = 1, isFiltrarTeams: bool = True,
                 isRecurrent: bool = True, funcAtiv: str = "softmax", isPassadaTempoDupla: bool = True,
                 qtdeDados: int = 40):

        # datasetEnt, datasetRot, datasetPrev = self.obterDatasetTeste(isSequencial=True)
        self.qtdeAcertos = []
        isApredizadoProfudo = False
        datasetEnt, datasetRot, datasetPrev, datasetValidar = self.obterDatasetRecurrentForKeras(
            id_team_home=id_team_home, id_team_away=id_team_away, isAmbas=isAmbas, isAgruparTeams=isAgruparTeams,
            idTypeReturn=idTypeReturn, isFiltrarTeams=isFiltrarTeams, isRecurrent=isRecurrent, funcAtiv=funcAtiv,
            isPassadaTempoDupla=isPassadaTempoDupla, qtdeDados=qtdeDados)

        modelRnn = self.build_model(datasetEnt, datasetRot, isSequencial=isRecurrent, funcAtiv=funcAtiv,
                                    isFiltrarTeams=isFiltrarTeams)

        if isApredizadoProfudo:
            self.predit, self.qtdeAcertos = self.treinarComAprendizadoProfundo(
                modelRnn, datasetEntrada=datasetEnt, datasetRortulo=datasetRot, datasetPrever=datasetPrev)
        else:
            self.predit, self.qtdeAcertos = self.treinarComAprendizadoNormal(model=modelRnn, datasetEntrada=datasetEnt,
                                                                             datasetRortulo=datasetRot,
                                                                             datasetPrever=datasetPrev,
                                                                             isFiltrarTeams=isFiltrarTeams,
                                                                             datasetValidar=datasetValidar,
                                                                             funcAtiv=funcAtiv,
                                                                             isRecurent=isRecurrent,
                                                                             qtdeDados=qtdeDados)

    def treinarComAprendizadoProfundo(self, model, datasetEntrada, datasetRortulo, datasetPrever):
        qtdeDadosAvalidacao = 10
        qtdeAcertos = [0, 0, 0, 0]

        newDatasetEntrada = []
        newDatasetRotulo = []

        for indexRemove in range(qtdeDadosAvalidacao):
            entradaRemovido = datasetEntrada.pop()
            rotuloRemovido = datasetRortulo.pop()
            newDatasetEntrada.insert(0, entradaRemovido)
            newDatasetRotulo.insert(0, rotuloRemovido)

        model.summary()
        numerActions = len(datasetRortulo[0])
        memory = SequentialMemory(limit=1000, window_length=1)
        policy = EpsGreedyQPolicy()
        dqn = DQNAgent(model=model, nb_actions=numerActions, memory=memory, nb_steps_warmup=100,
                       target_model_update=100, policy=policy, processor=CustomProcessor(), gamma=0.7, batch_size=1)

        footEnv = FootEnv(datasetEntrada=numpy.array(datasetEntrada),
                          datasetRotulo=numpy.array(datasetRortulo))
        funcLoss = "categorical_crossentropy"
        dqn.compile(optimizer=Adam(learning_rate=0.001), metrics=["accuracy"], loss=funcLoss)
        dqn = self.treinarRedeDQN(dqn=dqn, env=footEnv, nSteps=1000)

        for indexDadoVal in range(len(newDatasetEntrada)):
            previsaoValidar = numpy.array(model.predict(x=numpy.array([newDatasetEntrada[indexDadoVal]]))[0])
            indexPrev = numpy.argmax(previsaoValidar)
            indexRotu = numpy.argmax(newDatasetRotulo[indexDadoVal])
            indexEmpate = int(len(datasetRortulo[0]) / 2)

            if indexPrev == indexRotu:
                qtdeAcertos[0] += 1
            elif indexPrev == indexEmpate and indexRotu != indexEmpate:
                qtdeAcertos[1] += 1
            elif indexPrev != indexEmpate and indexRotu == indexEmpate:
                qtdeAcertos[2] += 1
            else:
                qtdeAcertos[3] += 1

            datasetEntrada.append(newDatasetEntrada[indexDadoVal])
            datasetRortulo.append(newDatasetRotulo[indexDadoVal])

            footEnv = FootEnv(datasetEntrada=numpy.array(datasetEntrada),
                              datasetRotulo=numpy.array(datasetRortulo))
            dqn = self.treinarRedeDQN(dqn=dqn, env=footEnv, nSteps=500)

        predit = model.predict(x=numpy.array(datasetPrever))
        predit = numpy.array(predit).tolist()
        return predit, qtdeAcertos

    def treinarRedeDQN(self, dqn: DQNAgent, env: FootEnv, nSteps: int = 2500):
        nDoWhile = 0
        isBreakWhile = False
        maxWhile = 2
        while not isBreakWhile and nDoWhile < maxWhile:
            dqn.fit(env=env, nb_steps=nSteps, visualize=False, verbose=2)

            history = dqn.test(env=env, nb_episodes=1, visualize=False, verbose=2)
            if history.history['episode_reward'][0] == len(env.datasetEntrada):
                isBreakWhile = True

            nDoWhile += 1

        return dqn

    def treinarComAprendizadoNormal(self, model: Sequential, datasetEntrada: list, datasetRortulo: list, datasetPrever,
                                    isFiltrarTeams: bool, datasetValidar: list, funcAtiv: str = "softmax",
                                    isRecurent: bool = True, qtdeDados: int = 40):
        funcStopping = EarlyStopping(monitor="accuracy", patience=250, mode="max")
        funcLoss = "categorical_crossentropy" if funcAtiv == "softmax" else "binary_crossentropy"
        qtdeDadosAvalidacao = 5 if isFiltrarTeams else 15
        if funcAtiv == "softmax":
            qtdeAcertos = [0, 0, 0]
        else:
            qtdeAcertos = [0 for i in range(len(datasetRortulo[0]))]
            qtdeAcertos.append(0)
        model.compile(optimizer=Adam(learning_rate=0.001), metrics=["accuracy"], loss=funcLoss)

        newDatasetEntrada = []
        newDatasetRotulo = []

        for indexRemove in range(qtdeDadosAvalidacao):
            entradaRemovido = datasetEntrada.pop()
            rotuloRemovido = datasetRortulo.pop()
            newDatasetEntrada.insert(0, entradaRemovido)
            newDatasetRotulo.insert(0, rotuloRemovido)

        model.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRortulo), callbacks=[funcStopping],
                  epochs=1000 if isFiltrarTeams else 750, batch_size=int(qtdeDados * 0.25),
                  validation_data=datasetValidar)

        for indexDadoVal in range(len(newDatasetEntrada)):
            previsaoValidar = numpy.array(model.predict(x=numpy.array([newDatasetEntrada[indexDadoVal]]))[0])
            indexPrev = numpy.argmax(previsaoValidar)
            indexRotu = numpy.argmax(newDatasetRotulo[indexDadoVal])
            indexEmpate = int(len(datasetRortulo[0]) / 2)

            if funcAtiv == "softmax":
                if indexPrev == indexRotu:
                    qtdeAcertos[0] += 1
                elif indexPrev == indexEmpate and indexRotu != indexEmpate:
                    qtdeAcertos[1] += 1
                elif indexPrev != indexEmpate and indexRotu == indexEmpate:
                    qtdeAcertos[1] += 1
                else:
                    qtdeAcertos[2] += 1
            else:
                for indx in range(len(previsaoValidar)):
                    valueNorm = 1 if previsaoValidar[indx] >= 0.5 else 0
                    if valueNorm == newDatasetRotulo[indexDadoVal][indx]:
                        qtdeAcertos[indx] += 1
                qtdeAcertos[-1] += 1
            datasetEntrada.append(newDatasetEntrada[indexDadoVal])
            datasetRortulo.append(newDatasetRotulo[indexDadoVal])

            '''model.reset_states()
            model.reset_metrics()'''
            funcStopping = EarlyStopping(monitor="accuracy", patience=75, mode="max")
            model.fit(x=numpy.array(datasetEntrada), y=numpy.array(datasetRortulo), callbacks=[funcStopping],
                      epochs=500, batch_size=int(qtdeDados * 0.25))

        predit = model.predict(x=numpy.array(datasetPrever))
        predit = numpy.array(predit).tolist()
        return predit, qtdeAcertos

    @staticmethod
    def build_model(datasetEntrada, datasetRotulo, isSequencial: bool = False, isCompile=True,
                    funcAtiv: str = 'softmax', isFiltrarTeams: bool = True):
        if isSequencial:
            isLSTM = False
            model = Sequential()
            if isLSTM:
                model.add(LSTM(units=int(64), activation="tanh",
                               input_shape=(len(datasetEntrada[0]), len(datasetEntrada[0][0])),
                               return_sequences=True))
                model.add(LSTM(units=int(128), activation="tanh"))
            else:
                model.add(SimpleRNN(units=int(128), activation="tanh",
                                    input_shape=(len(datasetEntrada[0]), len(datasetEntrada[0][0]))))
            model.add(Dense(units=len(datasetRotulo[0]), activation=funcAtiv))

            '''if isCompile:
                funcLoss = "categorical_crossentropy"
                model.compile(optimizer=Adam(learning_rate=0.01), metrics=["accuracy"], loss=funcLoss)'''
        else:
            model = Sequential()
            if isFiltrarTeams:
                model.add(Dense(units=len(datasetEntrada[0]), activation="tanh", input_shape=(len(datasetEntrada[0]),)))
            else:
                model.add(Dense(units=524, activation="tanh", input_shape=(len(datasetEntrada[0]),)))
            model.add(Dense(units=len(datasetRotulo[0]), activation=funcAtiv))

        model.summary()
        return model

    def obterDatasetRecurrentForKeras(self, id_team_home: int, id_team_away: int = None,
                                      isAmbas: bool = True, isAgruparTeams: bool = True,
                                      idTypeReturn: int = 1, isFiltrarTeams: bool = True, isRecurrent: bool = True,
                                      funcAtiv: str = "softmax", isPassadaTempoDupla: bool = True, qtdeDados: int = 40):
        funcAtivacao = funcAtiv

        arrIdsTeamSelec = [id_team_home]
        if isAmbas and id_team_away is not None:
            arrIdsTeamSelec.append(id_team_away)

        datasetPartidaRegras = DatasetPartidasRegras()
        datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(
            arrIdsTeam=arrIdsTeamSelec, isNormalizarSaidaEmClasse=funcAtivacao == "softmax",
            isFiltrarTeams=isFiltrarTeams, qtdeDados=qtdeDados, isAgruparTeams=isAgruparTeams,
            idTypeReturn=idTypeReturn, isPassadaTempoDupla=isPassadaTempoDupla, limitHistoricoMedias=2)

        if len(datasetEntrada) < qtdeDados:
            raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada)))
        else:
            print("Vai ser somente com " + str(len(datasetEntrada)) + " com idType: ", idTypeReturn)

        entradas, rotulos, prever, datasetValidar = (
            self.obterDatasetEntradaForRecurrent(entrada=datasetEntrada, rotulos=datasetRotulo, prever=datasetPrever,
                                                 qtdeTimesForTime=1, isRecurrent=isRecurrent,
                                                 isPassadaTempoDupla=isPassadaTempoDupla, qtdedadosValidar=0))

        return entradas, rotulos, prever, datasetValidar


    def obterDatasetTeste(self, isSequencial: bool = True):
        entrada = [
            [[1, 0, 0, 0, 0, 0]],
            [[0, 1, 0, 0, 0, 0]],
            [[0, 0, 1, 0, 0, 0]],
            [[0, 0, 0, 1, 0, 0]]
        ]

        rotulos = [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ]

        prever = [
            [[0, 0, 0, 0, 1, 0]]
        ]

        return entrada, rotulos, prever

    def obterDatasetEntradaForRecurrent(self, entrada, prever, rotulos, qtdeTimesForTime, isRecurrent,
                                        qtdedadosValidar: int = 0, isPassadaTempoDupla: bool = True):
        indexInit = 0
        arrNormalizadoEntrada = []
        arrNormalizadoRotulos = []
        arrNormalizadoPrever = []
        arrTimesEntrada = []
        arrTimesPrever = []
        datasetValidar = None

        if isPassadaTempoDupla:
            idAnterior = 0
            for iEnt in range(len(entrada)):
                arrTimesEntrada.append(entrada[iEnt])
                valValidar = entrada[iEnt][2]
                if (iEnt + 1) % 2 == 0:
                    if valValidar != idAnterior:
                        raise Exception("parece o id jogo anterior e", idAnterior, " e o proximo é ", valValidar)
                    arrNormalizadoEntrada.append(arrTimesEntrada)
                    arrNormalizadoRotulos.append(rotulos[iEnt - 1][0])
                    arrTimesEntrada = []

                else:
                    idAnterior = valValidar

            for iPrev in range(len(prever)):
                arrTimesPrever.append(prever[iPrev])

            if len(arrTimesPrever) != 2:
                raise Exception("Prever com ", len(arrTimesPrever), " e tem q ser 2")

            arrNormalizadoPrever.append(arrTimesPrever)

            datasetValidar = None
            if qtdedadosValidar >= 1:
                datasetValidar = (numpy.array(arrNormalizadoEntrada[-3:]), numpy.array(arrNormalizadoRotulos[-3:]))
                arrNormalizadoEntrada = arrNormalizadoEntrada[:-3]
                arrNormalizadoRotulos = arrNormalizadoRotulos[:-3]

            return arrNormalizadoEntrada, arrNormalizadoRotulos, arrNormalizadoPrever, datasetValidar
        else:
            if qtdeTimesForTime >= 2:

                for indexDado in range(len(entrada) - qtdeTimesForTime):
                    for iEnt in range(len(entrada)):
                        if iEnt < indexInit:
                            pass
                        else:
                            arrTimesEntrada.append(entrada[iEnt])

                            if len(arrTimesEntrada) == qtdeTimesForTime:
                                arrNormalizadoRotulos.append(rotulos[iEnt][0])
                                arrNormalizadoEntrada.append(arrTimesEntrada)
                                indexInit += 1
                                arrTimesEntrada = []
                                break

                arrNormalizadoPrever = []
                arrTimesPrever = arrNormalizadoEntrada[-1]
                arrTimesPrever.pop(0)
                arrTimesPrever.append(prever[0])
                arrNormalizadoPrever.append(arrTimesPrever)

                return arrNormalizadoEntrada, arrNormalizadoRotulos, arrNormalizadoPrever, datasetValidar
            else:
                arrEntradas = []
                arrPrever = []

                arrRotulos = []
                for i in range(len(entrada)):
                    if isRecurrent:
                        arrEntradas.append([entrada[i]])
                    else:
                        arrEntradas.append(entrada[i])

                    arrRotulos.append(rotulos[i][0])

                for i in range(len(prever)):
                    if isRecurrent:
                        arrPrever.append([prever[i]])
                    else:
                        arrPrever.append(prever[i])

                if qtdedadosValidar >= 1:
                    datasetValidar = (numpy.array(arrEntradas[-qtdedadosValidar:]), numpy.array(arrRotulos[-qtdedadosValidar:]))
                    arrNormalizadoEntrada = arrEntradas[:-qtdedadosValidar]
                    arrNormalizadoRotulos = arrRotulos[:-qtdedadosValidar]
                    return arrNormalizadoEntrada, arrNormalizadoRotulos, arrPrever, datasetValidar

                return arrEntradas, arrRotulos, arrPrever, datasetValidar




import numpy
import random

from keras.layers import Dense, LSTM, Input, concatenate, SimpleRNN
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping

from api.regras.datasetPartidasRegras import DatasetPartidasRegras
from api.regras.iaUteisRegras import IAUteisRegras


class KerasNeurais:

    @staticmethod
    def treinarLSTM(id_team_home: int, id_team_away: int = None, qtdeDados=60, isAmbas: bool = True,
                    isGols: bool = True, isAgruparTeams: bool = True, idTypeReturn: int = 1,
                    isFiltrarTeams: bool = True, isRecurrent: bool = True, funcAtiv: str = "softmax"):
        isReccurent = isRecurrent
        n_folds = 1 if isGols else 1
        isUnirDados = True
        qtdeDadosValidarPorFold = 1
        msgRetorno = "Não previu nada"
        uteisRegras = IAUteisRegras()
        funcAtivacao = funcAtiv  # "softmax"
        funcLoss = "categorical_crossentropy" if funcAtiv == "softmax" else "binary_crossentropy"   # "categorical_crossentropy"

        arrIdsTeamSelec = [id_team_home]
        if isAmbas:
            arrIdsTeamSelec.append(id_team_away)

        for nTentativa in range(3):
            patience = 1500 if isAgruparTeams else 1500
            if isFiltrarTeams:
                qtdeDados = 30 if isAmbas else 15
                nEpocas = 800 if isAmbas else 800
            else:
                qtdeDados = 430 if isAgruparTeams else 860
                nEpocas = 5000 if isAgruparTeams else 5500
            datasetPartidaRegras = DatasetPartidasRegras()
            datasetEntrada, datasetRotulo, datasetPrever = datasetPartidaRegras.obter(arrIdsTeam=arrIdsTeamSelec,
                                                                                      isNormalizarSaidaEmClasse=
                                                                                      funcAtivacao == "softmax",
                                                                                      isFiltrarTeams=isFiltrarTeams,
                                                                                      qtdeDados=qtdeDados,
                                                                                      isAgruparTeams=isAgruparTeams,
                                                                                      isForFF=False,
                                                                                      isDadosSoUmLado=False,
                                                                                      isGols=isGols,
                                                                                      idTypeReturn=idTypeReturn)

            if len(datasetEntrada) < qtdeDados:
                raise Exception("Sem dado suficientes temos somente: " + str(len(datasetEntrada)))
            else:
                print("Vai ser somente com " + str(len(datasetEntrada)))
                print("idType: ", idTypeReturn, " tentativa: ", nTentativa)

            entradas = []
            rotulos = []
            prever = []
            for i in range(len(datasetEntrada)):
                if isReccurent:
                    entradas.append([datasetEntrada[i]])
                else:
                    entradas.append(datasetEntrada[i])

                rotulos.append(datasetRotulo[i][0])

            for i in range(len(datasetPrever)):
                if isReccurent:
                    prever.append([datasetPrever[i]])
                else:
                    prever.append(datasetPrever[i])

            entradas = entradas
            rotulos = rotulos
            prever = prever

            nNeuroniosEntrada = len(entradas[0][0]) if isReccurent else len(entradas[0])

            inputL = Input(shape=(1 if isReccurent else None, nNeuroniosEntrada))
            qtdeSaida = len(rotulos[0])
            if isReccurent:
                ocultaLOne = SimpleRNN(units=int(nNeuroniosEntrada), activation="tanh")(inputL)
                # ocultaLTwo = SimpleRNN(units=int(nNeuroniosEntrada * 0.2), activation="tanh")(ocultaLOne)
                # ocultaLThree = LSTM(units=int(nNeuroniosEntrada * 0.5), activation="tanh")(ocultaLTwo)  # , return_sequences=True
                saidaL = Dense(units=qtdeSaida, name="saidaLOne", activation=funcAtivacao)(ocultaLOne)
            else:
                ocultaLOne = Dense(units=int(nNeuroniosEntrada * 0.5), activation="tanh")(inputL)
                # ocultaLTwo = Dense(units=int(nNeuroniosEntrada * 0.2), activation="tanh")(ocultaLOne)
                # ocultaLThree = Dense(units=int(nNeuroniosEntrada * 0.36), activation="tanh")(ocultaLTwo)  # , return_sequences=True
                saidaL = Dense(units=qtdeSaida, name="saidaLOne", activation=funcAtivacao)(ocultaLOne)

            model = Model(inputs=inputL, outputs=saidaL)
            model.compile(loss=funcLoss, optimizer='adam', metrics="accuracy")

            entradas_nFolds = uteisRegras.obter_k_folds_temporal(entradas, n_folds)
            rotulos_nFolds = uteisRegras.obter_k_folds_temporal(rotulos, n_folds)

            isAcertouAllFolds = True
            entradaFold = []
            rotuloFold = []
            for iFolds in range(len(entradas_nFolds)):
                print("idType: ", idTypeReturn, " tentativa: ", nTentativa, "fold: ", iFolds)
                isAcertouRotulo = False
                isValidarFolds = iFolds < len(entradas_nFolds) - 1

                if not isUnirDados:
                    entradaFold = []
                    rotuloFold = []

                for iDado in range(len(entradas_nFolds[iFolds])):
                    entradaFold.append(entradas_nFolds[iFolds][iDado])
                    rotuloFold.append(rotulos_nFolds[iFolds][iDado])

                entradasValidar = []
                rotulosValidar = []

                if isValidarFolds:
                    for v in range(qtdeDadosValidarPorFold):
                        entradaRemovida = entradaFold.pop()
                        rotuloRemovido = rotuloFold.pop()
                        entradasValidar.append(entradaRemovida)
                        rotulosValidar.append(rotuloRemovido)

                    entradasValidar = entradasValidar[::-1]
                    rotulosValidar = rotulosValidar[::-1]

                funcStopping = EarlyStopping(monitor="accuracy", patience=patience, mode="max")
                model.fit(entradaFold, rotuloFold, epochs=nEpocas, callbacks=[funcStopping],
                          batch_size=int(3))
                isAccuracyBoa = model.get_metrics_result()["accuracy"] >= 0.98

                if isValidarFolds:
                    if isUnirDados:
                        for iVal in range(len(entradasValidar)):
                            entradaFold.append(entradasValidar[iVal])
                            rotuloFold.append(rotulosValidar[iVal])
                    # Faz a previsão aqui.
                    previsao = model.predict(entradasValidar)

                    for indexPrev in range(len(previsao)):
                        if funcAtivacao == "softmax":
                            indexMaxPrevisao = numpy.argmax(previsao[indexPrev])
                            indexMaxRotulo = numpy.argmax(rotulosValidar[indexPrev])

                            if indexMaxPrevisao == indexMaxRotulo:
                                isAcertouRotulo = True

                        elif funcAtivacao == "sigmoid":
                            for iRot in range(len(previsao[indexPrev])):
                                throbleshot = 0.5
                                y = int(previsao[indexPrev][iRot] >= throbleshot)

                                if y == rotulosValidar[indexPrev][iRot]:
                                    isAcertouRotulo = True

                if (not isAcertouRotulo and isValidarFolds) or not isAccuracyBoa:
                    isAcertouAllFolds = False
                    break

            if not isAcertouAllFolds:
                continue

            previsaoDado = model.predict(prever)
            if len(previsaoDado) >= 1:
                msgRetorno = ""
            for i in range(len(previsaoDado)):
                strNorm = "["
                for j in range(len(previsaoDado[i])):
                    strNorm += str(j) + ": " + "{:.2f}".format(previsaoDado[i][j]) + ";  "
                strNorm += "]"

                if isGols:
                    msgRetorno += "As probabilidades são: " + strNorm + "\n"
                else:
                    msgRetorno += "As probabilidades são: " + strNorm + "\n"

            break

        return msgRetorno


# KerasNeurais.treinarLSTM(id_team_home=3912, isGols=False, isAmbas=False)

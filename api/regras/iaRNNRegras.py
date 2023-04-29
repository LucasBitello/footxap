from __future__ import annotations

import math
import numpy

from matplotlib import pyplot

from api.regras.iaRegras import IARegras, DatasetRNN

class RNN:
    def __init__(self, nNeuroniosEntrada: int, nNeuroniosCamadaOculta: list[int], nNeuroniosSaida: int, txAprendizado: float = None):
        self.iaRegras = IARegras()
        self.nNeuroniosEntrada = nNeuroniosEntrada
        self.nNeuroniosCamadaOculta = nNeuroniosCamadaOculta
        self.nNeuroniosSaida = nNeuroniosSaida
        self.txAprendizado = txAprendizado

        self.matriz_U: list[numpy.ndarray] = []
        self.matriz_W: list[numpy.ndarray] = []
        self.matriz_V: list[numpy.ndarray] = []
        self.matriz_B: list[numpy.ndarray] = []


        self.matriz_adagrad_U: list[numpy.ndarray] = []
        self.matriz_adagrad_W: list[numpy.ndarray] = []
        self.matriz_adagrad_V: list[numpy.ndarray] = []
        self.matriz_adagrad_B: list[numpy.ndarray] = []


        for indexnNeuroniosOcultos in range(len(nNeuroniosCamadaOculta)):
            W = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                     numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                     (nNeuroniosCamadaOculta[indexnNeuroniosOcultos],
                                      nNeuroniosCamadaOculta[indexnNeuroniosOcultos]))

            if indexnNeuroniosOcultos == 0:
                U = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         (nNeuroniosCamadaOculta[0], nNeuroniosEntrada))
            else:
                U = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         (nNeuroniosCamadaOculta[indexnNeuroniosOcultos],
                                          nNeuroniosCamadaOculta[indexnNeuroniosOcultos - 1]))

            self.matriz_adagrad_W.append(numpy.zeros_like(W))
            self.matriz_adagrad_U.append(numpy.zeros_like(U))

            #B = numpy.zeros((nNeuroniosCamadaOculta[indexnNeuroniosOcultos], 1))

            B = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                 numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                 (nNeuroniosCamadaOculta[indexnNeuroniosOcultos], 1))

            self.matriz_adagrad_B.append(numpy.zeros_like(B))

            self.matriz_B.append(B)
            self.matriz_U.append(U)
            self.matriz_W.append(W)

            if indexnNeuroniosOcultos == len(nNeuroniosCamadaOculta) - 1:
                V = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         (nNeuroniosSaida, nNeuroniosCamadaOculta[indexnNeuroniosOcultos]))

                self.matriz_V = V
                self.matriz_adagrad_V = numpy.zeros_like(V)

                B = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.nNeuroniosSaida)),
                                         (nNeuroniosSaida, 1))

                self.matriz_B.append(B)
                self.matriz_adagrad_B.append(numpy.zeros_like(B))

    def sigmoid(self, x):
        sig = 1 / (1 + numpy.exp(-x, dtype=numpy.float64))
        return sig

    def derivada_sigmoid(self, x):
        dsig = x * (1 - x)
        return dsig

    def softmax(self, x):
        exp_puntuacao = numpy.exp(x, dtype=numpy.float64)
        return exp_puntuacao / numpy.sum(exp_puntuacao, axis=0, dtype=numpy.float64)

    def derivada_softmax(self, x):
        s = self.softmax(x)
        return numpy.diagflat(s) - numpy.dot(s.T, s)

    def relu(self, x):
        saida = numpy.maximum(0, x)
        return saida

    def derivada_relu(self, x):
        saida = numpy.where(x > 0, 1, 0)
        return saida

    def derivada_tanh(self, estado_oculto: numpy.ndarray) -> numpy.ndarray:
        derivada_tanh = 1 - estado_oculto ** 2
        return derivada_tanh

    def forward(self, entradas: list[numpy.ndarray]):
        arrSaidas = []
        arrEstadosOcultos = []

        for i in range(len(entradas)):
            arrEstadosOcultosEntrada = []
            for j in range(len(self.nNeuroniosCamadaOculta)):
                arrEstadosOcultosEntrada.append(numpy.zeros((self.nNeuroniosCamadaOculta[j], 1)))
            arrEstadosOcultos.append(arrEstadosOcultosEntrada)

        for indexEntrada in range(len(entradas)):
            entrada = entradas[indexEntrada].reshape((self.nNeuroniosEntrada, 1))
            entrada_t = entrada

            for indexCamadaOculta in range(len(self.nNeuroniosCamadaOculta)):
                if indexCamadaOculta == 0:
                    entrada_t = entrada_t
                    estado_oculto_t = numpy.zeros((self.nNeuroniosCamadaOculta[indexCamadaOculta], 1))
                else:
                    entrada_t = arrEstadosOcultos[indexEntrada][indexCamadaOculta - 1]
                    estado_oculto_t = arrEstadosOcultos[indexEntrada][indexCamadaOculta]

                dot_U = numpy.dot(self.matriz_U[indexCamadaOculta], entrada_t)
                dot_W = numpy.dot(self.matriz_W[indexCamadaOculta], estado_oculto_t)
                sum_dot_U_W = dot_U + dot_W + self.matriz_B[indexCamadaOculta]
                estado_oculto_t = numpy.tanh(sum_dot_U_W)

                arrEstadosOcultos[indexEntrada][indexCamadaOculta] = estado_oculto_t
            saida_t = self.sigmoid(numpy.dot(self.matriz_V, arrEstadosOcultos[indexEntrada][-1]) + self.matriz_B[-1])
            arrSaidas.append(saida_t)

        return arrEstadosOcultos, arrSaidas

    def backward(self, entradas: list, esperado: list, saidas: list, estadosOcultos: list):
        lambda_reg = 0.005
        delta_U = [numpy.zeros((len(u), len(u[0]))) for u in self.matriz_U]
        delta_W = [numpy.zeros((nOculta, nOculta)) for nOculta in self.nNeuroniosCamadaOculta]
        delta_V = numpy.zeros((self.nNeuroniosSaida, self.nNeuroniosCamadaOculta[-1]))
        delta_B = [numpy.zeros((len(nBias), len(nBias[0]))) for nBias in self.matriz_B]

        for index_entrada_t in range(len(entradas) - 1, -1, -1):
            erro_t = saidas[index_entrada_t] - esperado[index_entrada_t]
            #atualiza o bias da saida
            delta_B[-1] += erro_t
            self.matriz_adagrad_B[-1] += delta_B[-1] ** 2
            self.matriz_B[-1] -= (self.txAprendizado * delta_B[-1]) / \
                                                  (numpy.sqrt(self.matriz_adagrad_B[-1]) + 1e-9)

            #delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_relu(saidas[index_entrada_t])
            delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_sigmoid(saidas[index_entrada_t])
            #delta_V += numpy.dot(self.derivada_softmax(saidas[index_entrada_t]), erro_t) * estadosOcultos[index_entrada_t][-1].T
            delta_oculto = numpy.dot(self.matriz_V.T, erro_t) * self.derivada_tanh(estadosOcultos[index_entrada_t][-1])

            for index_camada_oculta in range(len(self.nNeuroniosCamadaOculta) - 1, -1, -1):
                # atualiza o bias da ultima camada oculta.
                delta_B[index_camada_oculta] += delta_oculto

                if index_camada_oculta == 0:
                    delta_W[index_camada_oculta] += numpy.outer(delta_oculto, estadosOcultos[index_entrada_t][index_camada_oculta])
                    delta_U[index_camada_oculta] += numpy.outer(delta_oculto, entradas[index_entrada_t])
                    delta_oculto = entradas[index_entrada_t]

                else:
                    delta_W[index_camada_oculta] += numpy.outer(delta_oculto, estadosOcultos[index_entrada_t][index_camada_oculta])
                    delta_U[index_camada_oculta] += numpy.outer(delta_oculto, estadosOcultos[index_entrada_t][index_camada_oculta - 1])
                    delta_oculto = numpy.dot(self.matriz_U[index_camada_oculta].T, delta_oculto) * \
                                   self.derivada_tanh(estado_oculto=estadosOcultos[index_entrada_t][index_camada_oculta - 1])

                delta_W_regularized = delta_W[index_camada_oculta] + lambda_reg * self.matriz_W[index_camada_oculta]
                self.matriz_adagrad_W[index_camada_oculta] += self.matriz_W[index_camada_oculta] ** 2
                self.matriz_W[index_camada_oculta] -= (self.txAprendizado * self.matriz_W[index_camada_oculta]) / \
                                                      (numpy.sqrt(self.matriz_adagrad_W[index_camada_oculta]) + 1e-9)

                delta_U_regularized = delta_U[index_camada_oculta] + lambda_reg * self.matriz_U[index_camada_oculta]
                self.matriz_adagrad_U[index_camada_oculta] += delta_U_regularized ** 2
                self.matriz_U[index_camada_oculta] -= (self.txAprendizado * delta_U_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_U[index_camada_oculta]) + 1e-9)

                delta_B_regularized = delta_B[index_camada_oculta] + lambda_reg * self.matriz_B[index_camada_oculta]
                self.matriz_adagrad_B[index_camada_oculta] += self.matriz_B[index_camada_oculta] ** 2
                self.matriz_B[index_camada_oculta] -= (self.txAprendizado * self.matriz_B[index_camada_oculta]) / \
                                                      (numpy.sqrt(self.matriz_adagrad_B[index_camada_oculta]) + 1e-9)

            self.matriz_adagrad_V += delta_V ** 2
            self.matriz_V -= (self.txAprendizado * delta_V) / \
                             (numpy.sqrt(self.matriz_adagrad_V) + 1e-9)



    def treinar(self, entradas_treino: list[numpy.ndarray], saidas_treino: list[numpy.ndarray], n_epocas: int,
                tx_aprendizado: float, kFolds: int):
        self.txAprendizado = tx_aprendizado
        nKFolds =  kFolds
        saidas_treino = [numpy.asarray(rotulo).reshape(-1, 1) for rotulo in saidas_treino]
        arrLoss = numpy.zeros(25).tolist()

        epoch = 0
        indexLoss = 0
        isChegouPerto = False
        isBrekarWhile = False
        while not isBrekarWhile:
            epoch += 1
            entrada_in_k_folds = self.iaRegras.obter_k_folds_temporal(entradas_treino, nKFolds)
            saidas_in_k_folds = self.iaRegras.obter_k_folds_temporal(saidas_treino, nKFolds)
            #entrada_in_k_folds = [entradas_treino[:(len(entradas_treino) - 4)], entradas_treino[-4:]]
            #saidas_in_k_folds = [saidas_treino[:(len(saidas_treino) - 4)], saidas_treino[-4:]]
            if epoch == 0:
                """print(nKFolds, len(entrada_in_k_folds))
                print([len(k) for k in entrada_in_k_folds])
                print(sum([len(k) for k in entrada_in_k_folds]), len(entradas_treino))"""

            index_k_validation = len(entrada_in_k_folds) - 1

            for index_K_entrada in range(len(entrada_in_k_folds)):

                if index_K_entrada != index_k_validation:
                    estados_ocultos, saidas = self.forward(entradas=entrada_in_k_folds[index_K_entrada])
                    self.backward(entrada_in_k_folds[index_K_entrada], saidas_in_k_folds[index_K_entrada], saidas,
                                  estados_ocultos)
                else:
                    estados_ocultos, saidas = self.forward(entradas=entrada_in_k_folds[index_K_entrada])
                    self.backward(entrada_in_k_folds[index_K_entrada], saidas_in_k_folds[index_K_entrada], saidas,
                                  estados_ocultos)
                    estados_ocultos, saidas = self.forward(entradas=entrada_in_k_folds[index_K_entrada])
                    #mean absolute error" (MAE) ou "erro médio absoluto"
                    #loss = -numpy.mean(numpy.abs(numpy.asarray(saidas) - numpy.asarray(saidas_treino)))

                    #entropia cruzada (cross-entropy)
                    #loss = -numpy.mean(numpy.sum(saidas_treino * numpy.log(saidas), axis=0))

                    #(MSE - Mean Squared Error)
                    loss = numpy.mean((numpy.asarray(saidas_in_k_folds[index_K_entrada]) - numpy.asarray(saidas)) ** 2)
                    arrLoss[indexLoss] = loss
                    indexLoss += 1

                    if indexLoss >= len(arrLoss):
                        indexLoss = 0
                    else:
                        if arrLoss[-1] > 0.0:
                            if loss <= 0.04:
                                print("isChegou media")
                                isBrekarWhile = True
                            elif epoch >= n_epocas:
                                print("Brekou por epocas")
                                isBrekarWhile = True
                            #elif mediaLoss - arrLoss[0] <= 0.0001:
                                #self.txAprendizado = self.txAprendizado - (self.txAprendizado * 0.4)
                                #print("taxa de aprendizado alterada para: ", self.txAprendizado)

                    if (epoch + 1) % 5 == 0:
                        print("Epoch: ", epoch, ", erro: ", loss, "TxAprendizado: ", self.txAprendizado)
                        #self.txAprendizado = self.txAprendizado - (self.txAprendizado * 0.001)
                    else:
                        if n_epocas <= 25:
                            print("Epoch: ", epoch, ", erro: ", loss, "TxAprendizado: ", self.txAprendizado)

    def prever(self, entrada, isSaida = False):
        estados_ocultos, saida = self.forward(entrada)
        saida = [numpy.asarray(saida).reshape(-1)]
        saida_formatada = [f"{x:.7f}" for x in numpy.asarray(saida).reshape(-1)]
        print(saida_formatada)
        return estados_ocultos, saida

    def treinarRNN(self, datasetRNN: DatasetRNN):
        qtdeDados = len(datasetRNN.arr_entradas_treino)
        qtdeNeuroniosPrimeiraCamada = (int((qtdeDados * 2) + int(qtdeDados * 0.3)))
        taxaAprendizado = 0.002 if qtdeDados <= 115 else 0.01 if qtdeDados <= 196 else 0.008
        nKfolds = int(qtdeDados / 10) if (qtdeDados / 10) < int(qtdeDados / 10) + 0.5 else math.ceil(qtdeDados / 10)

        print("N neuronios entrada:", len(datasetRNN.arr_entradas_treino[0]))
        print("N neuronios primeira camada: ", qtdeNeuroniosPrimeiraCamada)
        print("Qtde dados:", qtdeDados, ", TTxAprendizado: ", taxaAprendizado, ", k-Folds:", nKfolds)

        self.__init__(nNeuroniosEntrada=len(datasetRNN.arr_entradas_treino[0]),
                      nNeuroniosCamadaOculta=[qtdeNeuroniosPrimeiraCamada,
                                              int(qtdeNeuroniosPrimeiraCamada * 0.7)],
                      nNeuroniosSaida=len(datasetRNN.arr_saidas_esperadas[0]))

        print(len(datasetRNN.arr_entradas_treino), len(datasetRNN.arr_saidas_esperadas))

        self.treinar(entradas_treino=datasetRNN.arr_entradas_treino, saidas_treino=datasetRNN.arr_saidas_esperadas,
                     n_epocas=500, tx_aprendizado=taxaAprendizado, kFolds=nKfolds)

        print(datasetRNN.dado_exemplo)
        print(datasetRNN.max_value_entradas, datasetRNN.min_value_entradas)
        print(datasetRNN.max_value_esperados, datasetRNN.min_value_esperados)

        for dadosPrever in datasetRNN.arr_prevevisao:
            print("Prever: ", dadosPrever)
            self.prever(entrada=[dadosPrever])

        return
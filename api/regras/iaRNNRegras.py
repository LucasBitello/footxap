from __future__ import annotations

import math
import numpy

from matplotlib import pyplot

from api.regras.iaRegras import IARegras

class DatasetRNN:
    def __init__(self):
        self.arr_entradas_treino: list = []
        self.arr_saidas_esperadas: list = []
        self.arr_prevevisao: list = []

        self.max_value_entradas: list = []
        self.min_value_entradas: list = []

        self.max_value_esperados: list = []
        self.min_value_esperados: list = []

        self.arr_name_values_entrada: list[str] = []
        self.arr_name_values_saida: list[str] = []

        self.dado_exemplo: any = None
        self.quantia_dados: int = None
        self.quantia_neuronios_entrada: int = None
        self.quantia_neuronios_saida: int = None


        if len(self.arr_entradas_treino) != len(self.arr_saidas_esperadas):
            raise "Datasets incompletos"

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
        exp_puntuacao = numpy.exp(x - numpy.max(x))
        soft = exp_puntuacao / numpy.sum(exp_puntuacao)
        return soft

    def derivada_softmax(self, x):
        s = x #self.softmax(x)
        deriv = numpy.diag(s) - numpy.outer(s, s.T)
        return deriv

    def derivada_softmax_matriz(self, x):
        m, n = x.shape
        dydx = numpy.zeros((m, n))
        for j in range(n):
            dydx[:, j] = numpy.diagonal(self.derivada_softmax(x[:, j]))
        return dydx

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
            saida_t = self.softmax(numpy.dot(self.matriz_V, arrEstadosOcultos[indexEntrada][-1]) + self.matriz_B[-1])
            arrSaidas.append(saida_t)

        return arrEstadosOcultos, arrSaidas

    def backward(self, entradas: list, esperado: list, saidas: list, estadosOcultos: list):
        lambda_reg = 0.01
        delta_U = [numpy.zeros((len(u), len(u[0]))) for u in self.matriz_U]
        delta_W = [numpy.zeros((nOculta, nOculta)) for nOculta in self.nNeuroniosCamadaOculta]
        delta_V = numpy.zeros((self.nNeuroniosSaida, self.nNeuroniosCamadaOculta[-1]))
        delta_B = [numpy.zeros((len(nBias), len(nBias[0]))) for nBias in self.matriz_B]

        for index_entrada_t in range(len(entradas) - 1, -1, -1):
            erro_t = saidas[index_entrada_t] - esperado[index_entrada_t]

            #atualiza o bias da saida
            derivda = self.softmax(saidas[index_entrada_t])
            delta_B[-1] += numpy.dot(erro_t.T, derivda)
            self.matriz_adagrad_B[-1] += delta_B[-1] ** 2
            self.matriz_B[-1] -= (self.txAprendizado * delta_B[-1]) / \
                                                  (numpy.sqrt(self.matriz_adagrad_B[-1]) + 1e-9)

            #delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_relu(saidas[index_entrada_t])
            #delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_sigmoid(saidas[index_entrada_t])
            delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_softmax_matriz(saidas[index_entrada_t])
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
                self.matriz_W[index_camada_oculta] -= (self.txAprendizado * delta_W_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_W[index_camada_oculta]) + 1e-9)

                delta_U_regularized = delta_U[index_camada_oculta] + lambda_reg * self.matriz_U[index_camada_oculta]
                self.matriz_adagrad_U[index_camada_oculta] += delta_U_regularized ** 2
                self.matriz_U[index_camada_oculta] -= (self.txAprendizado * delta_U_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_U[index_camada_oculta]) + 1e-9)

                delta_B_regularized = delta_B[index_camada_oculta] + lambda_reg * self.matriz_B[index_camada_oculta]
                self.matriz_adagrad_B[index_camada_oculta] += self.matriz_B[index_camada_oculta] ** 2
                self.matriz_B[index_camada_oculta] -= (self.txAprendizado * delta_B_regularized) / \
                                                      (numpy.sqrt(self.matriz_adagrad_B[index_camada_oculta]) + 1e-9)

            self.matriz_adagrad_V += delta_V ** 2
            self.matriz_V -= (self.txAprendizado * delta_V) / \
                             (numpy.sqrt(self.matriz_adagrad_V) + 1e-9)



    def treinar(self, entradas_treino: list[list[list]], saidas_treino: list[list[list]], n_epocas: int,
                tx_aprendizado: float):
        self.txAprendizado = tx_aprendizado
        epoch = 0
        isBrekarWhile = False

        while not isBrekarWhile:
            epoch += 1

            for index in range(len(entradas_treino)):
                estados_ocultos, saidas = self.forward(entradas=entradas_treino[index])

                if epoch % 2 == 0:
                    isBrekarWhile, mensagem = self.iaRegras.obterErroSaida(rotulos_saidas=saidas_treino[index],
                                                                           saidas_previstas=saidas, epoca_atual=epoch,
                                                                           taxa_aprendizado=self.txAprendizado)

                if epoch == n_epocas:
                    isBrekarWhile = True

                self.backward(entradas_treino[index], saidas_treino[index], saidas,
                              estados_ocultos)

    def prever(self, entrada, isSaida = False, isNormalizarSaida = True):
        estados_ocultos, saida = self.forward(entrada)
        saida = [numpy.asarray(saida).reshape(-1)]
        if isNormalizarSaida:
            saida_formatada = [f"{x*100:.4f}%" for x in numpy.asarray(saida).reshape(-1)]
        else:
            saida_formatada = [x for x in numpy.asarray(saida).reshape(-1)]
        return estados_ocultos, saida_formatada

    def treinarRNN(self, datasetRNN: DatasetRNN, isTreinar: bool = True) -> list[list]:
        if not isTreinar:
            return [[]]

        nEpocas = 100
        qtdeDados = datasetRNN.quantia_dados
        qtdeNeuroniosEntrada = datasetRNN.quantia_neuronios_entrada
        qtdeneuroniosSaida = datasetRNN.quantia_neuronios_saida
        qtdeNeuroniosPrimeiraCamada = 150
        taxaAprendizado = 0.001

        print("N neuronios entrada:", qtdeNeuroniosEntrada)
        print("N neuronios primeira camada oculta: ", qtdeNeuroniosPrimeiraCamada)
        print("Qtde dados:", qtdeDados, ", TxAprendizado: ", taxaAprendizado)


        self.__init__(nNeuroniosEntrada=qtdeNeuroniosEntrada,
                      nNeuroniosCamadaOculta=[int(qtdeNeuroniosPrimeiraCamada * 1.0),
                                              int(qtdeNeuroniosPrimeiraCamada * 0.7),
                                              int(qtdeNeuroniosPrimeiraCamada * 0.3)],
                      nNeuroniosSaida=qtdeneuroniosSaida)

        self.treinar(entradas_treino=datasetRNN.arr_entradas_treino, saidas_treino=datasetRNN.arr_saidas_esperadas,
                     n_epocas=nEpocas, tx_aprendizado=taxaAprendizado)

        print(datasetRNN.dado_exemplo)
        print(datasetRNN.max_value_esperados, datasetRNN.min_value_esperados)

        arrPrevisoes = []

        for dadosPrever in datasetRNN.arr_prevevisao:
            arrPrevisoes.append(self.prever(entrada=[dadosPrever])[1])

        return arrPrevisoes
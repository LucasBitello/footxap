from __future__ import annotations

import math
import numpy

from matplotlib import pyplot

from api.regras.iaRegras import IARegras

class CamadaSaidaRNN:
    def __init__(self, name_funcAtivacao: str, arr_saidas_esperas: list[list], qtde_neuronios: int):
        self.name_funcAtivacao = name_funcAtivacao
        self.funcAtivacao = None
        self.derivFuncAtivaccao = None
        self.arr_saidas_esperas: list[list] = arr_saidas_esperas
        self.qtde_neuronios: int = qtde_neuronios

        if self.name_funcAtivacao == "softmax":
            self.funcAtivacao = self.softmax
            self.derivFuncAtivaccao = self.derivada_softmax
        elif self.name_funcAtivacao == "sigmoid":
            self.funcAtivacao = self.sigmoid
            self.derivFuncAtivaccao = self.derivada_sigmoid
        elif self.name_funcAtivacao == "relu":
            self.funcAtivacao = self.relu
            self.funcAtivacao = self.derivada_relu
        else:
            raise "Renhuma função de ativação disponivel com esse nome: " + name_funcAtivacao
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
        s = x  # self.softmax(x)
        deriv = numpy.diag(s) - numpy.outer(s, numpy.transpose(s))
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


class DatasetRNN:
    def __init__(self):
        self.arr_prevevisao: list[list] = [[]]
        self.arr_entradas_treino: list[list] = [[]]
        self.arr_camadas_saidas: list[DatasetCamadaSaidaRNN] = [[]]

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


        if (self.arr_camadas_saidas is not None and self.arr_entradas_treino is not None) and\
                len(self.arr_entradas_treino) != len(self.arr_camadas_saidas):
            raise "Datasets incompletos"

class RNN:
    def __init__(self, nNeuroniosEntrada: int, nNeuroniosCamadaOculta: list[int],
                 arrCamadasSaida: list[CamadaSaidaRNN], txAprendizado: float = None):
        self.iaRegras = IARegras()
        self.nNeuroniosEntrada = nNeuroniosEntrada
        self.nNeuroniosCamadaOculta = nNeuroniosCamadaOculta
        self.arrCamadasSaida = arrCamadasSaida
        self.txAprendizado = txAprendizado
        self.funcAtivacaoCamadasOcultas = numpy.tanh
        self.derivFuncAtivacaoCamadasOcultas = self.derivada_tanh

        self.qtdeNeuroniosSaida = sum([camadaSaida.qtde_neuronios for camadaSaida in arrCamadasSaida])

        self.matriz_U: list[numpy.ndarray] = []
        self.matriz_W: list[numpy.ndarray] = []
        self.matriz_V: list[numpy.ndarray] = []
        self.matriz_B: list[numpy.ndarray] = []
        self.matriz_B_saida: list[numpy.ndarray] = []


        self.matriz_adagrad_U: list[numpy.ndarray] = []
        self.matriz_adagrad_W: list[numpy.ndarray] = []
        self.matriz_adagrad_V: list[numpy.ndarray] = []
        self.matriz_adagrad_B: list[numpy.ndarray] = []
        self.matriz_adagrad_B_saida: list[numpy.ndarray] = []


        for indexnNeuroniosOcultos in range(len(nNeuroniosCamadaOculta)):
            W = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                     numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                     (nNeuroniosCamadaOculta[indexnNeuroniosOcultos],
                                      nNeuroniosCamadaOculta[indexnNeuroniosOcultos]))

            if indexnNeuroniosOcultos == 0:
                U = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                         (nNeuroniosCamadaOculta[0], nNeuroniosEntrada))
            else:
                U = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                         numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                         (nNeuroniosCamadaOculta[indexnNeuroniosOcultos],
                                          nNeuroniosCamadaOculta[indexnNeuroniosOcultos - 1]))

            self.matriz_adagrad_W.append(numpy.zeros_like(W))
            self.matriz_adagrad_U.append(numpy.zeros_like(U))

            #B = numpy.zeros((nNeuroniosCamadaOculta[indexnNeuroniosOcultos], 1))

            B = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                     numpy.sqrt(2 / (self.nNeuroniosEntrada + self.qtdeNeuroniosSaida)),
                                     nNeuroniosCamadaOculta[indexnNeuroniosOcultos])

            self.matriz_adagrad_B.append(numpy.zeros_like(B))

            self.matriz_B.append(B)
            self.matriz_U.append(U)
            self.matriz_W.append(W)

            if indexnNeuroniosOcultos == len(nNeuroniosCamadaOculta) - 1:
                arr_matriz_B_saida = []
                arr_matriz_adagrad_B_saida = []
                for camadaSaida in arrCamadasSaida:
                    V = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + camadaSaida.qtde_neuronios)),
                                             numpy.sqrt(2 / (self.nNeuroniosEntrada + camadaSaida.qtde_neuronios)),
                                             (camadaSaida.qtde_neuronios, nNeuroniosCamadaOculta[indexnNeuroniosOcultos]))

                    self.matriz_V.append(V)
                    self.matriz_adagrad_V.append(numpy.zeros_like(V))

                    B_saida = numpy.random.uniform(-numpy.sqrt(2 / (self.nNeuroniosEntrada + camadaSaida.qtde_neuronios)),
                                                   numpy.sqrt(2 / (self.nNeuroniosEntrada + camadaSaida.qtde_neuronios)),
                                                   camadaSaida.qtde_neuronios)

                    self.matriz_B_saida.append(B_saida)
                    self.matriz_adagrad_B_saida.append(numpy.zeros_like(B_saida))

    def derivada_tanh(self, estado_oculto: numpy.ndarray) -> numpy.ndarray:
        derivada_tanh = 1 - estado_oculto ** 2
        return derivada_tanh

    def forward(self, entradas: list[numpy.ndarray]):
        arrSaidas = []
        arrEstadosOcultos = []
        for i in range(len(entradas)):
            arrEstadosOcultosEntrada = []
            for j in range(len(self.nNeuroniosCamadaOculta)):
                arrEstadosOcultosEntrada.append(numpy.zeros(self.nNeuroniosCamadaOculta[j]))
            arrEstadosOcultos.append(arrEstadosOcultosEntrada)

        for indexEntrada in range(len(entradas)):
            entrada = entradas[indexEntrada]
            entrada_t = entrada

            for indexCamadaOculta in range(len(self.nNeuroniosCamadaOculta)):
                if indexCamadaOculta == 0:
                    entrada_t = entrada_t
                    estado_oculto_t = numpy.zeros(self.nNeuroniosCamadaOculta[indexCamadaOculta])
                else:
                    entrada_t = arrEstadosOcultos[indexEntrada][indexCamadaOculta - 1]
                    estado_oculto_t = arrEstadosOcultos[indexEntrada][indexCamadaOculta]

                dot_U = numpy.dot(self.matriz_U[indexCamadaOculta], entrada_t)
                dot_W = numpy.dot(self.matriz_W[indexCamadaOculta], estado_oculto_t)
                sum_dot_U_W = dot_U + dot_W + self.matriz_B[indexCamadaOculta]
                estado_oculto_t = self.funcAtivacaoCamadasOcultas(sum_dot_U_W)

                arrEstadosOcultos[indexEntrada][indexCamadaOculta] = estado_oculto_t

            arrCamadasSaidas_t = []

            for indexCamadaSaida in range(len(self.arrCamadasSaida)):
                camadaSaida = self.arrCamadasSaida[indexCamadaSaida]
                saida_camada_t = camadaSaida.funcAtivacao(numpy.dot(self.matriz_V[indexCamadaSaida], arrEstadosOcultos[indexEntrada][-1]) + self.matriz_B_saida[indexCamadaSaida])
                arrCamadasSaidas_t.append(saida_camada_t)

            arrSaidas.append(arrCamadasSaidas_t)

        return arrEstadosOcultos, arrSaidas

    def backward(self, entradas: list, esperado: list, saidas: list, estadosOcultos: list):
        lambda_reg = 0.01
        delta_U = [numpy.zeros((len(u), len(u[0]))) for u in self.matriz_U]
        delta_W = [numpy.zeros((nOculta, nOculta)) for nOculta in self.nNeuroniosCamadaOculta]
        delta_V = [numpy.zeros((camadaSaida.qtde_neuronios, self.nNeuroniosCamadaOculta[-1])) for camadaSaida in self.arrCamadasSaida]
        delta_B = [numpy.zeros(len(bias)) for bias in self.matriz_B]
        delta_B_saida = [numpy.zeros(len(nBias_saida)) for nBias_saida in self.matriz_B_saida]

        for index_entrada_t in range(len(entradas) - 1, -1, -1):
            arr_erros_t = []
            arr_deltas_ocultos = []
            delta_oculto = numpy.zeros(len(estadosOcultos[index_entrada_t][-1]))

            for index_camada_saida in range(len(self.arrCamadasSaida)):
                camadaSaida = self.arrCamadasSaida[index_camada_saida]
                erro_t = saidas[index_entrada_t][index_camada_saida] - camadaSaida.arr_saidas_esperas[index_entrada_t]
                arr_erros_t.append(erro_t)

                #atualiza o bias da saida
                derivda = camadaSaida.funcAtivacao(saidas[index_entrada_t][index_camada_saida])
                delta_B_saida[index_camada_saida] += numpy.dot(erro_t.T, derivda)
                self.matriz_adagrad_B_saida[index_camada_saida] += delta_B_saida[index_camada_saida] ** 2
                self.matriz_B_saida[index_camada_saida] -= (self.txAprendizado * delta_B_saida[index_camada_saida]) / \
                                                      (numpy.sqrt(self.matriz_adagrad_B_saida[index_camada_saida]) + 1e-9)

                # delta_V += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_relu(saidas[index_entrada_t])
                #delta_V[index_camada_saida] += numpy.dot(erro_t, estadosOcultos[index_entrada_t][-1].T) * self.derivada_sigmoid(saidas[index_entrada_t][index_camada_saida])
                #delta_V[index_camada_saida] += numpy.dot(numpy.dot(numpy.transpose([erro_t]), [estadosOcultos[index_entrada_t][-1]]),
                                                         #camadaSaida.derivFuncAtivaccao([saidas[index_entrada_t][index_camada_saida]]))
                delta_V[index_camada_saida] += numpy.dot(numpy.transpose([erro_t]), [estadosOcultos[index_entrada_t][-1]])

                self.matriz_adagrad_V[index_camada_saida] += delta_V[index_camada_saida] ** 2
                self.matriz_V[index_camada_saida] -= (self.txAprendizado * delta_V[index_camada_saida]) / \
                                 (numpy.sqrt(self.matriz_adagrad_V[index_camada_saida]) + 1e-9)


                delta_oculto += numpy.dot(self.matriz_V[index_camada_saida].T, erro_t) * self.derivFuncAtivacaoCamadasOcultas(estadosOcultos[index_entrada_t][-1])

            for index_camada_oculta in range(len(self.nNeuroniosCamadaOculta) - 1, -1, -1):
                # atualiza o bias da ultima camada oculta.
                delta_B[index_camada_oculta] += delta_oculto

                if index_camada_oculta == 0:
                    delta_W[index_camada_oculta] += numpy.multiply(delta_oculto, estadosOcultos[index_entrada_t][index_camada_oculta].T)
                    delta_U[index_camada_oculta] += numpy.dot(delta_oculto[:, numpy.newaxis], entradas[index_entrada_t][numpy.newaxis, :])
                    delta_oculto = entradas[index_entrada_t]

                else:
                    delta_W[index_camada_oculta] += numpy.multiply(delta_oculto, estadosOcultos[index_entrada_t][index_camada_oculta])
                    delta_U[index_camada_oculta] += numpy.dot(delta_oculto[:, numpy.newaxis], estadosOcultos[index_entrada_t][index_camada_oculta - 1][numpy.newaxis, :])
                    delta_oculto = numpy.dot(self.matriz_U[index_camada_oculta].T, delta_oculto) * \
                                   self.derivFuncAtivacaoCamadasOcultas(estado_oculto=estadosOcultos[index_entrada_t][index_camada_oculta - 1])

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

    def treinar(self, entradas_treino: list[list], saidas_treino: list[CamadaSaidaRNN], n_epocas: int,
                tx_aprendizado: float) -> float:
        self.txAprendizado = tx_aprendizado
        epoch = 0
        isBrekarWhile = False
        loss = 0
        while not isBrekarWhile:
            epoch += 1
            estados_ocultos, saidas = self.forward(entradas=entradas_treino)

            if epoch % 10 == 0 or epoch == 1:
                isBrekarWhile, mensagem, loss = self.obterErroSaidaRNN(rotulos_saidas=saidas_treino,
                                                                       saidas_previstas=saidas, epoca_atual=epoch,
                                                                       taxa_aprendizado=self.txAprendizado)

            if epoch == n_epocas:
                isBrekarWhile = True

            self.backward(entradas_treino, saidas_treino, saidas,estados_ocultos)

        return loss

    def obterErroSaidaRNN(self, rotulos_saidas: list[CamadaSaidaRNN], saidas_previstas: list, epoca_atual: int, taxa_aprendizado: float,
                       erro_aceitavel: float = 0.01, isPrintarMensagem: bool = True) -> list[bool, str]:
        # mean absolute error" (MAE) ou "erro médio absoluto"
        # loss = -numpy.mean(numpy.abs(numpy.asarray(saidas) - numpy.asarray(saidas_in_k_folds[index_entrada])))

        # entropia cruzada (cross-entropy)
        loss = 0
        for indexSaida in range(len(saidas_previstas)):
            for indexCamadaSaida in range(len(rotulos_saidas)):
                camadaSaida = rotulos_saidas[indexCamadaSaida]
                loss += -numpy.mean(numpy.sum(camadaSaida.arr_saidas_esperas[indexSaida] * numpy.log(saidas_previstas[indexSaida][indexCamadaSaida]), axis=0))

        # (MSE - Mean Squared Error)
        #loss = numpy.mean(numpy.power(numpy.asarray(rotulos_saidas) - numpy.asarray(saidas_previstas), 2))

        isBrekarTreino: bool = False

        if loss <= erro_aceitavel:
            isBrekarTreino = True

        mensagem_loss = f"Epoch: {epoca_atual}, erro: {loss}, TxAprendizado: {taxa_aprendizado}"

        if isPrintarMensagem:
            print(mensagem_loss)

        return isBrekarTreino, mensagem_loss, loss

    def prever(self, entrada, isSaida = False, isNormalizarSaida = True):
        estados_ocultos, camada_saidas_rede = self.forward(entrada)
        saidas_formatada = []

        for camada_saida in camada_saidas_rede:
            for saida in camada_saida:
                if isNormalizarSaida:
                    saida_formatada = [f"{x * 100:.4f}%" for x in numpy.asarray(saida).reshape(-1)]
                    saidas_formatada.append(saida_formatada)
                else:
                    saidas_formatada.append([x for x in numpy.asarray(saida).reshape(-1)])

        return estados_ocultos, saidas_formatada

    def treinarRNN(self, datasetRNN: DatasetRNN, isTreinar: bool = True,
                   nNeuroniosPrimeiraCamada: int = 100, nEpocas: int = 2000, txAprendizado: float = 0.01) -> list[list]:
        if not isTreinar:
            return [[]]

        nEpocas = nEpocas
        qtdeDados = datasetRNN.quantia_dados
        qtdeNeuroniosEntrada = datasetRNN.quantia_neuronios_entrada
        qtdeNeuroniosPrimeiraCamada = 200
        taxaAprendizado = txAprendizado

        print("N neuronios entrada:", qtdeNeuroniosEntrada)
        print("N neuronios primeira camada oculta: ", qtdeNeuroniosPrimeiraCamada)
        print("Qtde dados:", qtdeDados, ", TxAprendizado: ", taxaAprendizado)

        self.__init__(nNeuroniosEntrada=qtdeNeuroniosEntrada,
                      nNeuroniosCamadaOculta=[int(qtdeNeuroniosPrimeiraCamada * 1.0)],
                      arrCamadasSaida=datasetRNN.arr_camadas_saidas)

        loss = self.treinar(entradas_treino=datasetRNN.arr_entradas_treino, saidas_treino=datasetRNN.arr_camadas_saidas,
                            n_epocas=nEpocas, tx_aprendizado=taxaAprendizado)

        print(datasetRNN.dado_exemplo)
        print(datasetRNN.max_value_esperados, datasetRNN.min_value_esperados)

        arrPrevisoes = []

        for dadosPrever in datasetRNN.arr_prevevisao:
            arrPrevisoes.append(self.prever(entrada=[dadosPrever])[1])

        return arrPrevisoes, loss
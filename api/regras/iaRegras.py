from __future__ import annotations
from scipy.special import expit
import math
import numpy

from matplotlib import pyplot

class IARegras:
    def normalizar_dataset(self, dataset, max_valor: list = None, min_valor: list = None) -> tuple[list, list, list]:
        dataset = numpy.asarray(dataset)
        max_valor = numpy.amax(dataset, axis=0) if max_valor is None else max_valor
        min_valor = numpy.amin(dataset, axis=0) if min_valor is None else min_valor

        arrDividendos = max_valor - min_valor
        dividendos = []
        for dividendo in arrDividendos:
            if dividendo >= 1:
                dividendos.append(dividendo)
            else:
                dividendos.append(1)

        dataset_normalizado: list[list] = (dataset - min_valor) / dividendos
        return dataset_normalizado, max_valor, min_valor


    def obter_k_folds_temporal(self, arrDados: list, n_folds: int):
        len_k_folds = int(len(arrDados) / n_folds) + 1
        arrDadosCortados = list(arrDados)
        new_k_arr_dados = []
        new_k = []

        while len(arrDadosCortados) >= 1 :
            index = 0
            new_k.append(arrDadosCortados[index])
            arrDadosCortados.pop(index)

            if len(new_k) == len_k_folds:
                new_k_arr_dados.append(new_k)
                new_k = []

        if len(new_k) >= 1 and len(new_k_arr_dados) < n_folds:
            new_k_arr_dados.append(new_k)
        elif len(new_k) >= 1 and len(new_k_arr_dados) == n_folds:
            raise "Divisão dos dados errada"

        return new_k_arr_dados

    def softmax(self, x):
        exp_puntuacao = numpy.exp(x - numpy.max(x))
        soft = exp_puntuacao / numpy.sum(exp_puntuacao)
        return soft

    def normalizarRotulosEmClasses(self, arrRotulosOriginais: list[list[list]], max_value_rotulos: list) -> list[list[list]]:
        arrDadosRotulosNormalizado = []
        for rotulo in arrRotulosOriginais:
            camadas_saida = []
            for index_val_rotulo in range(len(rotulo)):
                camada_saida = numpy.zeros(max_value_rotulos[index_val_rotulo] + 1, dtype=numpy.int32)
                camada_saida[rotulo[index_val_rotulo]] = 1
                camadas_saida.append(camada_saida)
            arrDadosRotulosNormalizado.append(camadas_saida)

        return arrDadosRotulosNormalizado

    def derivada_softmax_matriz(self, x):
        s = x  # self.softmax(x)
        deriv = numpy.diag(s) - numpy.outer(s, numpy.transpose(s))
        return deriv

    def derivada_softmax(self, x):
        m, n = x.shape
        dydx = numpy.zeros((m, n))
        for j in range(n):
            dydx[:, j] = numpy.diagonal(self.derivada_softmax_matriz(x[:, j]))
        return dydx

    def tanh(self, x):
        return numpy.tanh(x)

    def derivada_tanh(self, x):
        derivada_tanh = 1 - x ** 2
        return derivada_tanh

    def sigmoid(self, x) -> list | float:
        #sig = 1 / (1 + numpy.exp(-x, dtype=numpy.float64))
        sig = expit(x)
        return sig

    def derivada_sigmoid(self, x):
        dsig = x * (1 - x)
        return dsig

    def inicalizarPesosXavier(self, nItens: int, tupleDim: tuple) -> list:
        arrXavier = numpy.random.uniform(-numpy.sqrt(2 / nItens), numpy.sqrt(2 / nItens), tupleDim)
        return arrXavier

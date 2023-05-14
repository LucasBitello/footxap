from __future__ import annotations

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

    def obterErroSaida(self, rotulos_saidas: list, saidas_previstas: list, epoca_atual: int, taxa_aprendizado: float,
                       erro_aceitavel: float = 0.01, isPrintarMensagem: bool = True) -> list[bool, str]:
        # mean absolute error" (MAE) ou "erro médio absoluto"
        # loss = -numpy.mean(numpy.abs(numpy.asarray(saidas) - numpy.asarray(saidas_in_k_folds[index_entrada])))

        # entropia cruzada (cross-entropy)
        loss = -numpy.mean(numpy.sum(rotulos_saidas * numpy.log(saidas_previstas), axis=0))

        # (MSE - Mean Squared Error)
        #loss = numpy.mean(numpy.power(numpy.asarray(rotulos_saidas) - numpy.asarray(saidas_previstas), 2))

        isBrekarTreino: bool = False

        if loss <= erro_aceitavel:
            isBrekarTreino = True

        mensagem_loss = f"Epoch: {epoca_atual}, erro: {loss}, TxAprendizado: {taxa_aprendizado}"

        if isPrintarMensagem:
            print(mensagem_loss)

        return isBrekarTreino, mensagem_loss
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


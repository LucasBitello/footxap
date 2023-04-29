class ParamsNotNone:
    def __init__(self):
        self.nameColumns: list[str] = []
        self.dataColumns: list = []

class UteisRegras():
    def retornarSomenteParamsNotNone(self, dictDados: dict) -> ParamsNotNone:
        paramsNotNone: ParamsNotNone = ParamsNotNone()
        for key in dictDados.keys():
            if dictDados[key] is not None:
                paramsNotNone.nameColumns.append(key)
                paramsNotNone.dataColumns.append(dictDados[key])

        return paramsNotNone

    def normalizarDadosForView(self, arrDados: list[object]) -> list[dict]:
        arrDadosJson = []
        arrDadosNormalizados = []

        for dado in arrDados:
            if (type(dado) != int and type(dado) != float and type(dado) != str and type(dado) != list and dado is not None):
                arrDadosJson.append(dado.__dict__)
            else:
                arrDadosNormalizados.append(dado)

        for dado in arrDadosJson:
            for key in dado.keys():
                if type(dado[key]) == list:
                    dado[key] = self.normalizarDadosForView(dado[key])
            arrDadosNormalizados.append(dado)

        return arrDadosNormalizados
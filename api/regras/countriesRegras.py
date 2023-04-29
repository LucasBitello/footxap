from api.models.countriesModel import CountriesModel, Country

class CountriesRegras:
    def __init__(self):
        self.countriesModel = CountriesModel()

    def obter(self, id: int = None) -> list[Country]:
        if id is None:
            self.countriesModel.atualizarDados()
            arrDados = self.countriesModel.obterTudo()
        else:
            arrDados = self.countriesModel.obterByColumnsID(arrDados=[id])

            if len(arrDados) == 1:
                country: Country = arrDados[0]
                country.is_obter_dados = 1
                self.countriesModel.salvar(data=[country])
                arrDados = [country]

        return arrDados
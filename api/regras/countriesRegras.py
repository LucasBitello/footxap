from api.models.countriesModel import CountriesModel, Country

class CountriesRegras:
    def __init__(self):
        self.countriesModel = CountriesModel()

    def obter(self, id: int = None) -> list:
        if id is None:
            self.countriesModel.atualizarDados()
            arrDados = self.countriesModel.obterTudo()
        else:
            arrDados = self.countriesModel.obterByColumnsID(arrDados=[id])

        return arrDados
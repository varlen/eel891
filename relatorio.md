## Entendimento do conjunto de dados

O conjunto de dados fornecido contempla 81 colunas com 1460 registros para teste e 1460 registros para treino. Utilizou-se a caracterização presente no arquivo data_description.txt para ter o significado de cada variável.

Para obter uma inferência quantitativa menos heuristica, inicialmente as variáveis numéricas foram utilizadas para criação de gráficos de espalhamento, possibilitando estimar a influência individual de cada uma destas.

OverallQual -> 0 a 10

MiscVal --> Pode ser uma caracteristica qualquer. Devo descontar do valor final?


---

----- LotShape vs LandContour

Ambas categorias parecem consideravelmente correlacionadas, porém LandContour parece mais regular.

Foi gerado um mapa de calor incluindo as 10 variáveis mais correlacionadas com o preço da venda. 

<img src="./imagens/hm.png">

O mapa de calor mostra que as variáveis GrLivArea e TotRmsAbvGrd, TotalBsmtSF e 1stFlrSF são altamente correlacionadas. Assim, TotalRmsAbvGrd e TotalBsmtSF foram desconsideradas. 

Através desta observação preliminar do conjunto de dados, foi feita a caracterização das diferentes variáveis disponíveis. 




Com intuito de simplificar o conjunto de dados, algumas colunas foram desconsideradas. São essas a coluna Street que dentro dos dados de treino sempre apresentou o valor "Pave", Utilities que com exceção de um registro, apresentou valor "AllPub", 
## Entendimento do conjunto de dados

O conjunto de dados fornecido contempla 81 colunas com 1460 registros para teste e 1460 registros para treino.
Inicialmente utilizou-se um software de planilha para observação dos dados em forma tabular. As variáveis também foram utilizadas para criação de gráficos de espalhamento, possibilitando estimar a influência individual de cada uma destas.

Através desta observação preliminar do conjunto de dados, foi feita a caracterização das diferentes variáveis disponíveis. 

<table>
    <thead>
        <th>Variável</th>
        <th>Tipo</th>
        <th>Categoria</th>
        <th>Influência Esperada</th>
        <th>Incluir?</th>
        <th>Obs</th>
    </thead>
    <tr>
        <td>MSSubClass</td>
        <td>Discreta</td>
        <td>-</td>
        <td>Baixa</td>
        <td>Não</td>
        <td>- </td>
    </tr>
    <tr>
        <td>MSZoning</td>
        <td>Discreta</td>
        <td>-</td>
        <td>Baixa</td>
        <td>Não</td>
        <td>1151 de 1460 entradas apresentam valor RL, com 5 valores distintos.</td>
    </tr>


</table>

Com intuito de simplificar o conjunto de dados, algumas colunas foram desconsideradas. São essas a coluna Street que dentro dos dados de treino sempre apresentou o valor "Pave", Utilities que com exceção de um registro, apresentou valor "AllPub", 
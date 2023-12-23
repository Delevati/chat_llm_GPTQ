# WizardLM-13B-V1-1-SuperHOT-8K-GPTQ

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/WizardLM-13B-V1.1-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Você é um programa para extrair campos importantes de um texto, eu preciso que vc extraia os campos: Numero da nota, data e hora de emissao como um unico campo, o município onde o serviço foi prestado, o codigo do serviço, o valor total bruto da nota, o CPF ou CNPJ do prestador e o CPF ou CNPJ do tomador, Campo Observacao/descriminacao de servico. Me de a resposta com apenas os campos extraidos em um formato json e a chave dos valors deve os nomes das tags sendo essas exatas: numeroNFSe, dataEmissaoNFSe, municipioExecServico, codigoServico, valorServico, CPF_CNPJ_Prestador, CPF_CNPJ_Tomador, Observacao."
prompt_template=f'''PREFEITURA MUNICIPAL DE CORRENTINA Data da Competência: SECRETARIA DE FINANÇAS Abril/2023 NOTA FISCAL DE SERVIÇOS ELETRÔNICA - NFS-e ata e Hora da Emissão: 10/04/2023 15:03:00 Código Verificação: D911D6C2E PRESTADOR DE SERVIÇOS CPF/CNPJ: Inscrição Municipal: 31.074.994/0001-14 335300116 Telefone: Inscrição Estadual: 61.34912647. Nome/Razão Social: Nome Fantasia: ; ERIKA CRISTINA CERQUEIRA DE |» -Sem Logomar LIMA 84401893115 LIDER DIESEL PECAS Endereço: AVN AVN RIO GRANDE DO SUL (19) Nº 07 BAIRRO CIDADE ROSARIO CIDADE: CORRENTINA - BA E-mail: ERIKACERQUEIRA30QGMAIL.COM TOMADOR DE SERVIÇOS CPF/CNPJ: Inscrição Municipal: 02150533000266 Telefone: Inscrição Estadual: Nome/Razão Social: SANTA CRUZ POWER CORPORATION USINAS HIDRELETRICAS S.A. Endereço: RUA ESTRADA DE SÃO DOMINGOS, S/N - KM 8,2 Nº S/N BAIRRO: ZONA RURAL CIDADE: SAO DOMINGOS - GO CEP: 73860000 E-mail: Não Informado DISCRIMINAÇÃO DOS SERVIÇOS SERVIÇO DE MANUTENÇÃOE PREVENTIVA NO GERADOR DIESEL - IRYNA. VALOR TOTAL DA NOTA: R$ 3.099,40 CNAE - 45.20-0/01 - SERVIÇOS DE MANUTEÇÃO E REPARAÇÃO MECÂNICA DE VEICULOS AUTOMOTORES. º Item da Lista de Serviços - 14.01 - LUBRIFICAÇÃO, LIMPEZA, LUSTRAÇÃO, REVISÃO, CARGA E RECARGA, CONSERTO, RESTAURAÇÃO, BLINDAGEM, MANUTENÇÃO E CONSERVAÇÃO DE MÁQUINAS, VEÍCULOS, APARELHOS, EQUIPAMENTOS, MOTORES, ELEVADORES OU DE QUALQUE VALOR SERVIÇOS: VALOR | DESC. INCOND: BASE DE ALÍQUOTA: VALOR ISS: VALOR ISS DESC. COND: [n] R$ 3.099,40 DEDUÇÃO: R$ 0,00 CÁLCULO: 0% R$ 0,00 RETIDO: R$ 0,00 R$ 0,00 R$ 3.099,40 R$ 0,00 VALOR PIS: VALOR COFINS: VALOR IR: VALOR INSS: VALOR CSLL: OUTRAS RETENÇÕES: VALOR LÍQUIDO: R$ 0,00 R$ 0,00 R$ 0,00 R$ 0,00 R$ 0,00 R$ 0,00 R$ 3.099,40 EXIGIBILIDADE ISS REGIME TRIBUTAÇÃO SIMPLES NACIONAL LOCAL. PRESTAÇÃO LOCAL INCIDÊNCIA ISS Retido Exigivel Microempresário Individual Sim (0% ) SERVIÇO CORRENTINA - BA Não (MEI) CORRENTINA - BA Observação: - PRESTADOR OPTANTE DO SIMPLES NACIONAL (ALÍQUOTA: 0 % USER: {prompt} ASSISTANT:

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

print(pipe(prompt_template)[0]['generated_text'])
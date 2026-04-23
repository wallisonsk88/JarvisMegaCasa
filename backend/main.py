from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading
import time
import json
import os
import webbrowser
import asyncio
import edge_tts
import base64
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import requests
import sqlite3
import winsound
import subprocess
import contextlib
import io
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
try:
    from bs4 import BeautifulSoup
except:
    BeautifulSoup = None

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
except:
    search_tool = None

try:
    import pyautogui
except:
    pyautogui = None

CONFIG_FILE = "config.json"
AGENDA_DB = "mega_agenda.db"

def init_db():
    conn = sqlite3.connect(AGENDA_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lembretes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT NOT NULL,
            data_hora TEXT NOT NULL,
            status TEXT DEFAULT 'pendente'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS atalhos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL UNIQUE,
            url TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memoria_longo_prazo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            informacao TEXT NOT NULL
        )
    """)
    # Links iniciais passados pelo Wallison
    atalhos_iniciais = [
        ('suporte', 'https://meganet-suport-git-main-wallison-rangels-projects.vercel.app/'),
        ('atlaz', 'https://meganett.atlaz.com.br/admin'),
        ('rede mega', 'https://meganett.atlaz.com.br/admin'),
        ('flash monitor', 'https://flashmonitor.com.br') # Exemplo se tiver
    ]
    for n, u in atalhos_iniciais:
        cursor.execute("INSERT OR IGNORE INTO atalhos (nome, url) VALUES (?, ?)", (n, u))
    conn.commit()
    conn.close()

init_db()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

event_queue = asyncio.Queue()

class ConfigModel(BaseModel):
    modelType: str
    apiKey: str
    systemEmail: str
    systemPassword: str
    sensitivity: float = 0.002

class ChatMessage(BaseModel):
    message: str

current_config = {
    "modelType": "groq", 
    "apiKey": "", 
    "systemEmail": "", 
    "systemPassword": "",
    "sensitivity": 0.002
}

async def gerar_audio_base64(texto):
    try:
        communicate = edge_tts.Communicate(texto, "pt-BR-AntonioNeural", rate="+0%", pitch="-5Hz")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        try: os.unlink(tmp_path)
        except: pass
        return b64
    except Exception as e:
        print(f"[TTS ERR] {e}")
        return None

def get_transcription(file_path, prompt="Mega"):
    """Transcreve áudio detectando o provedor pela chave ou configuração."""
    api_key = current_config.get("apiKey", "")
    m_type = current_config.get("modelType", "groq")
    
    if not api_key: return ""

    try:
        # 1. Tenta adivinhar o motor de áudio pela chave (mais robusto)
        is_groq = api_key.startswith("gsk_")
        is_gemini = api_key.startswith("AIza")
        is_together = api_key.startswith("top_")

        # Prioridade 1: Groq (Whisper V3 Turbo - Mais rápido)
        if is_groq or (m_type == "groq" and not is_gemini and not is_together):
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {api_key}"}
            with open(file_path, "rb") as f:
                resp = requests.post(url, headers=headers, files={"file": f}, 
                                     data={"model": "whisper-large-v3-turbo", "language": "pt", "prompt": prompt}, timeout=10)
                if resp.status_code == 200: return resp.json().get("text", "").strip()
        
        # Prioridade 2: Together AI
        elif is_together or (m_type == "together" and not is_gemini):
            url = "https://api.together.xyz/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {api_key}"}
            with open(file_path, "rb") as f:
                resp = requests.post(url, headers=headers, files={"file": f}, 
                                     data={"model": "whisper-1", "language": "pt", "prompt": prompt}, timeout=10)
                if resp.status_code == 200: return resp.json().get("text", "").strip()

        # Prioridade 3: Gemini (Multimodal)
        elif is_gemini or m_type == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            with open(file_path, "rb") as f:
                audio_data = f.read()
            response = model.generate_content([
                {"mime_type": "audio/wav", "data": audio_data},
                f"Transcreva este áudio em português. Retorne APENAS o texto. Contexto: {prompt}"
            ])
            return response.text.strip()
            
    except Exception as e:
        print(f"[TRANSCRIPT ERROR] Provedor {m_type} falhou: {e}")
        
    return ""


# --- FERRAMENTAS ---
def skill_pesquisar_web(query):
    if not search_tool: return "Módulo de pesquisa indisponível."
    try: return search_tool.run(query)
    except Exception as e: return f"Erro na pesquisa: {e}"

def skill_agenda_lembrete(texto, data_hora=""):
    try:
        conn = sqlite3.connect(AGENDA_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO lembretes (texto, data_hora) VALUES (?, ?)", (texto, data_hora))
        conn.commit(); conn.close()
        return f"Lembrete agendado: '{texto}'."
    except Exception as e: return f"Erro na agenda: {e}"

def skill_listar_agenda():
    try:
        conn = sqlite3.connect(AGENDA_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT texto, data_hora FROM lembretes WHERE status = 'pendente' LIMIT 5")
        tasks = cursor.fetchall(); conn.close()
        if not tasks: return "Sua agenda está vazia."
        res = "Seus lembretes:\n"
        for t in tasks: res += f"- {t[0]} ({t[1]})\n"
        return res
    except: return "Erro ao acessar agenda."

def skill_controlar_midia(acao):
    if not pyautogui: return "Mídia indisponível."
    try:
        if "pause" in acao or "play" in acao: pyautogui.press("playpause")
        elif "proxima" in acao or "skip" in acao: pyautogui.press("nexttrack")
        elif "anterior" in acao: pyautogui.press("prevtrack")
        return "Comando enviado."
    except: return "Erro mídia."

def skill_controlar_janela(acao, alvo=""):
    try:
        import pyautogui
        import subprocess
        
        acao = acao.lower()
        if acao == "minimizar_tudo":
            pyautogui.hotkey('win', 'd')
            return "Todas as janelas foram minimizadas."
        elif acao == "maximizar":
            pyautogui.hotkey('win', 'up')
            return "Janela atual maximizada."
        elif acao == "minimizar":
            pyautogui.hotkey('win', 'down')
            return "Janela atual minimizada."
        elif acao == "fechar_atual":
            pyautogui.hotkey('alt', 'f4')
            return "Janela fechada."
        elif acao == "fechar_programa" and alvo:
            subprocess.run(["taskkill", "/F", "/IM", f"{alvo}.exe"], capture_output=True)
            return f"Programa {alvo} encerrado."
        else:
            return "Ação não reconhecida."
    except Exception as e:
        return f"Erro de janela: {e}"

def skill_adicionar_atalho(nome, url):
    try:
        conn = sqlite3.connect(AGENDA_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO atalhos (nome, url) VALUES (?, ?)", (nome.lower(), url))
        conn.commit(); conn.close()
        return f"Link de {nome} gravado na memória base."
    except Exception as e: return f"Falha ao gravar memória: {e}"

def skill_extrair_conteudo(url):
    if not BeautifulSoup: return "Módulo de extração (bs4) indisponível."
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200: return f"Erro ao acessar site: Status {resp.status_code}"
        
        soup = BeautifulSoup(resp.text, 'lxml')
        # Remove lixo
        for s in soup(['script', 'style', 'nav', 'footer', 'header']): s.decompose()
        
        texto = soup.get_text(separator='\n')
        linhas = [l.strip() for l in texto.splitlines() if l.strip()]
        res = "\n".join(linhas)
        return res[:4000] # Limite para não estourar contexto
    except Exception as e:
        return f"Falha na extração direta: {e}"

def skill_salvar_memoria(info):
    try:
        conn = sqlite3.connect(AGENDA_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO memoria_longo_prazo (informacao) VALUES (?)", (info,))
        conn.commit(); conn.close()
        return f"Informação gravada na memória de longo prazo: {info}"
    except Exception as e: return f"Falha ao gravar memória: {e}"

def skill_abrir_atalho(nome, url_sugerido=""):
    try:
        # 1. Checa a memória principal do usuário
        conn = sqlite3.connect(AGENDA_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM atalhos WHERE nome LIKE ?", (f"%{nome.lower()}%",))
        res = cursor.fetchone(); conn.close()
        
        if res:
            webbrowser.open(res[0])
            return f"Abrindo {nome} direto da nossa memória."
        elif url_sugerido and str(url_sugerido).startswith("http"):
            # 2. IA encontrou o link publicamente
            webbrowser.open(url_sugerido)
            skill_adicionar_atalho(nome, url_sugerido) # Grava na memória
            return f"Não tinha na memória. Abri o site e já gravei o atalho '{nome}' para acessos futuros."
        else:
            # 3. Fallback inteligente (Google/Direct)
            termo = nome.lower().replace(" ", "")
            url_gerada = f"https://{termo}.com"
            webbrowser.open(url_gerada)
            skill_adicionar_atalho(nome, url_gerada) # Grava na memória
            return f"Tentando rota de acesso direto. Gravei o atalho '{nome}' na memória automaticamente."
    except Exception as e: return f"Erro ao acessar navegadores: {e}"

def skill_run_cmd(command):
    try:
        print(f"[SYSTEM EXEC] Terminal Rota Autorizada: {command[:200]}")
        resultado = subprocess.run(["powershell", "-Command", command], capture_output=True, text=True, timeout=40)
        out = resultado.stdout if resultado.stdout else resultado.stderr
        return out if out else "Comando powershell executado silenciosamente e com sucesso."
    except Exception as e:
        return f"Erro Crítico de Console: {e}"

def skill_python_exec(codigo):
    output = io.StringIO()
    try:
        print("[SYSTEM EXEC] Executando bloco Python interno.")
        with contextlib.redirect_stdout(output):
            exec(codigo, globals())
        val = output.getvalue()
        return val if val else "Script rodou com sucesso sem gerar prints na tela."
    except Exception as e:
        return f"Exceção no kernel Python: {e}"

class MegaAgent:
    def __init__(self, config):
        self.config = config
        m_type = str(config.get("modelType", "groq")).lower().strip()
        api_key = str(config.get("apiKey", "")).strip()
        
        print(f"[SISTEMA] Inicializando Kernel Cognitivo: {m_type.upper()}")
        
        try:
            if m_type == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0.2)
            elif m_type == "openrouter":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", model="meta-llama/llama-3.3-70b-instruct", temperature=0.2)
            elif m_type == "together":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(api_key=api_key, base_url="https://api.together.xyz/v1", model="meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature=0.2)
            else:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.2)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[CRITICAL ERR] Falha ao carregar Kernel {m_type}: {error_trace}")
            self.llm = None
            raise Exception(f"Erro no Kernel {m_type}: {str(e)}")
            
        self.history = []
        try:
            if os.path.exists("mega_history.json"):
                with open("mega_history.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                for msg in data:
                    if msg["type"] == "human": self.history.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "ai": self.history.append(AIMessage(content=msg["content"]))
                    elif msg["type"] == "system": self.history.append(SystemMessage(content=msg["content"]))
        except Exception as e: print("[HISTORY LOAD ERROR]", e)

    async def verify_kernel(self):
        """Testa se o kernel consegue responder um 'ping' básico."""
        if not self.llm: return False
        try:
            # Tenta um invoke rápido com timeout de 10s
            import asyncio
            await asyncio.wait_for(asyncio.to_thread(self.llm.invoke, "Responda apenas OK"), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            print("[VERIFY ERR] Timeout na resposta da IA.")
            raise Exception("A IA demorou muito para responder. Verifique sua internet ou a chave.")
        except Exception as e:
            print(f"[VERIFY ERR] {e}")
            return False

    def save_history(self):
        try:
            data = []
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            for msg in self.history:
                if isinstance(msg, HumanMessage): data.append({"type": "human", "content": str(msg.content)})
                elif isinstance(msg, AIMessage): data.append({"type": "ai", "content": str(msg.content)})
                elif isinstance(msg, SystemMessage): data.append({"type": "system", "content": str(msg.content)})
            with open("mega_history.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e: print("[HISTORY SAVE ERROR]", e)
    
    def get_memoria_links(self):
        try:
            conn = sqlite3.connect(AGENDA_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT nome FROM atalhos")
            links = [row[0] for row in cursor.fetchall()]
            conn.close()
            return ", ".join(links)
        except: return "Nenhum"

    def get_memoria_longo_prazo(self):
        try:
            conn = sqlite3.connect(AGENDA_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT informacao FROM memoria_longo_prazo")
            infos = [row[0] for row in cursor.fetchall()]
            conn.close()
            return " | ".join(infos)
        except: return "Nenhum"

    def process_command(self, command_text):
        if not hasattr(self, 'history'):
            self.history = []
            
        self.history.append(HumanMessage(content=command_text))
        
        # Limita histórico base para não estourar tokens
        if len(self.history) > 16:
            self.history = self.history[-16:]
            
        self.save_history()

        links_salvos = self.get_memoria_links()
        memorias_gerais = self.get_memoria_longo_prazo()
        
        # Variáveis Fixas de Sistema Real
        user_profile = os.environ.get("USERPROFILE", "C:\\Users")
        
        tentativas_de_loop = 0

        while True:
            tentativas_de_loop += 1
            if tentativas_de_loop > 6:
                return "Sr., demorei demais processando isso. Estou abortando para evitar falhas sistêmicas."
                
            prompt = f"""[SYSTEM] Você é o MEGA Executive, um Sistema Autônomo e IA de nível kernel do Wallison Rangel.
ESTADO ATUAL DO KERNEL: {self.config.get('modelType', 'UNKNOWN').upper()}
Você agora tem capacidade de OpenInterpreter: pode ler e executar comandos livremente no computador dele via Poweshell ou executar Python abstrato.
Caminho Oficial do Sistema do Usuário: {user_profile}  (ATENÇÃO: Este caminho contém espaços. SEMPRE use aspas no PowerShell, ex: mkdir "{user_profile}\\Desktop\\MegaA")
Links salvos: {links_salvos} | Fatos Importantes: {memorias_gerais}.

-> REGRAS DE AUTONOMIA E SEGURANÇA (LEIA COM ATENÇÃO): <-
Se você precisar CRIAR pastas, MODIFICAR arquivos, INSTALAR dependências ou EXCLUIR alguma coisa, VOCÊ DEVE OBRIGATORIAMENTE usar o skill "ask_permission" ANTES de executar o comando. Não rode o terminal antes da aprovação do Wallison.
Se ele te der o comando (ex: "pode fazer, crie a pasta"), no seu próximo turno você usará "run_cmd" e passará o comando no terminal.
Comandos de leitura (dir, ler arquivo com type, ping, consultar info base) ou execução de scripts Python não-destrutivos não precisam de permissão.

-> REGRAS DE AUTOMAÇÃO WEB (SELENIUM): <-
Se o usuário pedir para PREENCHER FORMULÁRIOS, LER CÓDIGO HTML ou NAVEGAR de forma complexa, use "run_python" e escreva um script com `selenium`.
1) Inicie o navegador e abra o site: `from selenium import webdriver; from selenium.webdriver.common.by import By; from selenium.webdriver.chrome.service import Service; from webdriver_manager.chrome import ChromeDriverManager; driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())); driver.get('URL')`
2) O interpretador persiste as variáveis na área local da memória (global). Ou seja, se você já criou o `driver` num turno anterior, NAS PRÓXIMAS EXECUÇÕES PODE APENAS USÁ-LO diretamente! **NUNCA CHAME webdriver.Chrome() COM O NAVEGADOR JÁ ABERTO!** Evite sobreposição. Se der erro NameError 'driver', inicie-o.
3) Para descobrir os campos ocultos: rode `print(driver.page_source)` para ler o HTML no retorno do Log, ou tente localizar imprimindo nomes/IDs.
4) Para interagir: rode um Python interagindo de fato: `driver.find_element(By.ID, 'x').send_keys('dado')` seguido de `driver.find_element(By.XPATH, 'xx').click()`.
5) NÃO faça `driver.quit()`, mantenha aberto pro usuário visualizar.

-> REGRAS DE PESQUISA E EXTRAÇÃO DE DADOS: <-
Sempre que o usuário pedir para buscar algo em um site, siga este protocolo:
1) Se for uma busca geral, use "search".
2) Se você tiver a URL ou encontrar uma URL relevante, use "extract_web" para ler o conteúdo real do site. É muito mais preciso que o resumo da busca.
3) Só use Selenium ("run_python") se o site precisar de cliques ou se for um SPA que não carrega conteúdo sem JS. Para leitura rápida de texto/código, use "extract_web".

-> LISTA DE SKILLS E FORMATO JSON OBRIGATÓRIO: <-
- CHAT: {{ "skill": "none", "response": "sua resposta final falada de forma curta..." }} (Use para falar e encerrar turno)
- ASK_PERMISSION: {{ "skill": "ask_permission", "command_intent": "criar app", "response": "Sr, preciso abrir o powershell para iniciar o app. Autoriza?" }} (Encerra o turno aguardando resposta)
- TERMINAL_CMD: {{ "skill": "run_cmd", "command": "comando powershell/cmd válido no windows usando aspas em caminhos com espaco" }} (Fica no loop invisível e lê o resultado no log)
- PYTHON_RUN: {{ "skill": "run_python", "code": "print('hello')" }} (Fica no loop invisível)
- BROWSER_OPEN_LINK: {{ "skill": "open_link", "name": "...", "url": "https://..." }} (Encerra turno com Acesso Concedido)
- SEARCH: {{ "skill": "search", "query": "..." }}
- EXTRACT_WEB: {{ "skill": "extract_web", "url": "https://..." }} (Use para ler conteúdo real de uma página)
- AGENDA_ADD: {{ "skill": "agenda_add", "text": "...", "time": "..." }}
- MEMORY_SAVE_FACT: {{ "skill": "save_fact", "fact": "..." }}
- MEMORY_SAVE_LINK: {{ "skill": "save_link", "name": "...", "url": "http..." }}
- MEDIA: {{ "skill": "media", "action": "pause|play|next|prev" }}
- WINDOW_CONTROL: {{ "skill": "window_control", "action": "minimizar_tudo|maximizar|minimizar|fechar_atual|fechar_programa", "target": "chrome (opcional)" }}

Horário Atual: {datetime.now().strftime('%H:%M:%S')}"""

            messages = [SystemMessage(content=prompt)] + self.history

            try:
                print(f"[MATRIZ] Processando via Kernel: {self.config.get('modelType', 'UNK').upper()} | LLM: {type(self.llm).__name__}")
                resp = self.llm.invoke(messages)
                self.history.append(resp) # Salva a própria saída para manter a coesão
                self.save_history() # Salva no disco
                
                raw_text = resp.content.strip()
                # Tenta extrair qualquer coisa entre chaves se houver Markdown
                import re
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = raw_text

                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as je:
                    print(f"[JSON FIX LOOP] Erro na formatação. Forçando auto-correção...")
                    self.history.append(SystemMessage(content="ATENÇÃO: Sua última resposta NÃO foi um JSON válido. Por favor, corrija e responda EXATAMENTE e APENAS no formato { \"skill\": \"...\" } sem textos adicionais."))
                    continue
                    
                skill = data.get("skill")
                
                print(f"[REASONING] A IA optou pela skill: '{skill}'")

                # -> Interruções de Turno (Devolvem áudio e pausam) <-
                if skill == "none":
                    return data.get("response", "Finalizado.")
                elif skill == "ask_permission":
                    return data.get("response", "Me dê permissão para rodar a rotina de terminal, senhor.")
                elif skill == "open_link":
                    res = skill_abrir_atalho(data.get("name"), data.get("url", ""))
                    return f"Portas abertas. {res}"
                elif skill == "save_fact":
                    return skill_salvar_memoria(data.get("fact"))
                elif skill == "save_link":
                    return skill_adicionar_atalho(data.get("name"), data.get("url"))
                elif skill == "media":
                    return skill_controlar_midia(data.get("action"))
                elif skill == "window_control":
                    return skill_controlar_janela(data.get("action"), data.get("target", ""))
                elif skill == "agenda_add":
                    return skill_agenda_lembrete(data.get("text"), data.get("time"))
                
                # -> Loop Invisível Autônomo (Injeta a resposta do comando de volta na história) <-
                elif skill == "run_cmd":
                    cmd = data.get("command", "")
                    out = skill_run_cmd(cmd)
                    self.history.append(SystemMessage(content=f"> [Powershell Output de `{cmd}`]:\n{out[:1200]}"))
                    # Deixa rodar o While novamente para que o modelo LEIA o output e tome próxima decisão
                    continue

                elif skill == "run_python":
                    code = data.get("code", "")
                    out = skill_python_exec(code)
                    self.history.append(SystemMessage(content=f"> [Python Output]:\n{out[:2000]}"))
                    # Continua o loop
                    continue

                elif skill == "search":
                    query = data.get("query", "")
                    out = skill_pesquisar_web(query)
                    self.history.append(SystemMessage(content=f"> [DuckDuckGo Output]:\n{out[:1200]}"))
                    # Continua o loop
                    continue

                elif skill == "extract_web":
                    url = data.get("url", "")
                    out = skill_extrair_conteudo(url)
                    self.history.append(SystemMessage(content=f"> [Site Content Output from {url}]:\n{out}"))
                    continue
                
                elif skill == "agenda_list":
                    out = skill_listar_agenda()
                    self.history.append(SystemMessage(content=f"> [Agenda Lida]:\n{out}"))
                    continue

                else:
                    return data.get("response", "Matriz instável. Comandos autônomos falharam.")

            except Exception as e:
                error_msg = str(e)
                print(f"[ERRO DE ROTA COGNITIVA] {error_msg}")
                
                if "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    return "Sr., sua chave de API parece estar inválida ou expirou para este modelo. Por favor, verifique as configurações."
                elif "rate_limit" in error_msg.lower():
                    return "Sr., atingimos o limite de requisições do provedor atual. Por favor, aguarde um momento."
                
                return f"Sr., ocorreu uma falha técnica no kernel {self.config.get('modelType')}: {error_msg[:100]}. Loop abortado."

mega_agent_instance = None

@app.post("/api/config")
async def save_config(config: ConfigModel):
    global mega_agent_instance
    print(f"[CONFIG] ATUALIZANDO CORE: {config.modelType.upper()}")
    try:
        # 1. Atualiza Configuração
        current_config.update(config.dict())
        
        # 2. Tenta Instanciar o Kernel
        new_agent = MegaAgent(current_config)
        
        # 3. Teste de Fogo (Verifica se a chave funciona)
        print(f"[CONFIG] TESTANDO CHAVE {config.modelType.upper()}...")
        verified = await new_agent.verify_kernel()
        if not verified:
            raise Exception(f"A chave API para {config.modelType} parece inválida ou o serviço está offline.")
            
        # 4. Se passou, assume o novo agente
        mega_agent_instance = new_agent
        with open(CONFIG_FILE, "w") as f: json.dump(current_config, f)
        
        # Bip de sucesso na troca
        try: winsound.Beep(1000, 100); winsound.Beep(1500, 100)
        except: pass
        
        msg = f"Kernel {config.modelType} validado e ativo."
        audio = await gerar_audio_base64(msg)
        return {"status": "success", "response": msg, "audio": audio, "model": config.modelType}
        
    except Exception as e:
        import traceback
        err_msg = str(e)
        print(f"[CONFIG ERR] {traceback.format_exc()}")
        return {"status": "error", "message": err_msg}

@app.get("/api/config")
def get_config(): 
    return {**current_config, "validated": mega_agent_instance is not None and mega_agent_instance.llm is not None}

@app.get("/api/calibrate")
async def calibrate():
    fs = 16000
    try:
        rec = sd.rec(int(2.0 * fs), samplerate=fs, channels=1); sd.wait()
        rms = np.sqrt(np.mean(rec**2))
        return {"suggested": max(0.001, float(rms * 2.5))}
    except: return {"suggested": 0.002}

@app.post("/api/chat")
async def chat(chat_msg: ChatMessage):
    global mega_agent_instance
    if not mega_agent_instance: return {"response": "Sem API Key."}
    res = mega_agent_instance.process_command(chat_msg.message)
    audio = await gerar_audio_base64(res)
    return {"response": res, "audio": audio}

@app.get("/api/events")
async def events(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected(): break
            try: yield f"data: {json.dumps(await event_queue.get())}\n\n"
            except: break
    return StreamingResponse(gen(), media_type="text/event-stream")

# --- MOTOR DE ESCUTA (TWO-STAGE WAKE) ---
def voice_listener_loop(loop):
    fs = 16000
    is_processing = False
    print("[SISTEMA] Motor de Despertar Ativo. Aguardando 'MEGA'...")

    while True:
        threshold = current_config.get("sensitivity", 0.002)
        if not (mega_agent_instance and current_config["apiKey"]) or is_processing:
            time.sleep(0.5); continue
        
        try:
            # Abre o stream uma única vez fora do sub-loop
            try:
                stream = sd.InputStream(channels=1, samplerate=fs)
                stream.start()
            except Exception as e:
                print(f"[MOTOR] Erro ao abrir microfone: {e}")
                time.sleep(5); continue

            print("[SISTEMA] Motor de Despertar Online. Diga 'MEGA' para começar.")
            
            while True:
                threshold = current_config.get("sensitivity", 0.002)
                if is_processing:
                    time.sleep(0.5); continue
                
                # ETAPA 1: Monitoramento Silencioso (Buffer de 2 segundos)
                # Usamos read com block=False se possível, ou apenas lemos o chunk
                data, _ = stream.read(int(1.5 * fs)) 
                vol = np.linalg.norm(data) / np.sqrt(len(data))
                
                if vol > threshold:
                    print(f"[DEBUG] Som detectado (Vol: {vol:.4f}) - Analisando...")
                    
                    # Captura rápida...
                    fd, p1 = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                    wavfile.write(p1, fs, data)
                    
                    text = get_transcription(p1, prompt="Mega").lower()
                    os.unlink(p1)
                    
                    if not text and current_config.get("modelType") == "openrouter":
                        print("[AVISO] OpenRouter não possui serviço de voz. Use uma chave Groq ou Gemini para me ouvir.")
                        time.sleep(2)
                        continue

                    if text:
                        print(f"[ESCUTA] {text}")
                        wake_words = ["mega", "meiga", "meca", "nega", "amiga", "hey", "még", "vegas", "mika", "neca", "brega"]
                        
                        if any(w in text for w in wake_words):
                            print("[!] MEGA DESPERTO. Ouvindo comando...")
                            is_processing = True
                            asyncio.run_coroutine_threadsafe(event_queue.put({"type": "wake_detected"}), loop)
                            try: winsound.Beep(800, 150); winsound.Beep(1200, 150)
                            except: pass
                            
                            # Inicializa o tempo da última fala
                            ultima_fala = time.time()
                            
                            # Loop de Conversação Contínua
                            ultima_interacao = time.time()
                            while True:
                                # Escuta o comando por 4 segundos (mais ágil que 8s)
                                try:
                                    # Verifica timeout de inatividade (2 minutos)
                                    if time.time() - ultima_interacao > 120:
                                        print("[SISTEMA] Inatividade detectada (2min). Hibernando...")
                                        break

                                    cmd_data, _ = stream.read(int(4.0 * fs))
                                    vol_cmd = np.linalg.norm(cmd_data) / np.sqrt(len(cmd_data))
                                    
                                    # Se estiver muito silêncio, nem gasta API de transcrição
                                    if vol_cmd < (threshold * 0.8):
                                        time.sleep(0.1)
                                        continue

                                    fd, p2 = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                                    wavfile.write(p2, fs, cmd_data)
                                    
                                    comando = get_transcription(p2, prompt="Mega, comando, ação, pesquisar, sistema.").strip()
                                    os.unlink(p2)
                                except Exception as loop_err:
                                    print(f"[LOOP ERR] {loop_err}")
                                    continue
                                    
                                if len(comando) < 3:
                                    continue
                                    
                                ultima_interacao = time.time()
                                print(f"[USUÁRIO] {comando}")

                                # Verifica comando manual de desativação
                                cmd_clean = comando.lower()
                                desativar_keywords = ["desativar", "dormir", "ficar quieta", "pode ir", "encerrar", "desligar sistema", "tchau mega", "até logo", "descansar"]
                                if any(kw in cmd_clean for kw in desativar_keywords):
                                    print("[SISTEMA] Hibernando via comando.")
                                    msg = "Entendido, senhor. Estou entrando em modo de espera. Basta me chamar pelo nome quando precisar."
                                    mega_agent_instance.history.append(AIMessage(content=msg))
                                    audio_b64 = asyncio.run_coroutine_threadsafe(gerar_audio_base64(msg), loop).result()
                                    asyncio.run_coroutine_threadsafe(event_queue.put({"type": "voice_response", "text": msg, "audio": audio_b64}), loop)
                                    break
                                
                                ultima_interacao = time.time()
                                res_text = mega_agent_instance.process_command(comando)
                                audio_b64 = asyncio.run_coroutine_threadsafe(gerar_audio_base64(res_text), loop).result()
                                asyncio.run_coroutine_threadsafe(event_queue.put({"type": "voice_response", "text": res_text, "audio": audio_b64}), loop)
                                
                                # Aguarda o tempo aproximado da fala da MEGA para não gravar a própria voz (feedback loop)
                                tempo_fala = len(res_text.split()) / 2.2 # ~2.2 palavras por segundo
                                
                                # Pausa a gravação do microfone para evitar erro de buffer overflow (PortAudio) durante a fala
                                stream.stop()
                                time.sleep(tempo_fala + 1.0)
                                
                                # Bip curto para avisar que está pronta para ouvir novamente
                                try: winsound.Beep(1000, 150)
                                except: pass
                                
                                # Retoma a gravação do microfone
                                stream.start()
                                
                                # Avisa o frontend que está pronta e gravando de novo
                                asyncio.run_coroutine_threadsafe(event_queue.put({"type": "wake_detected"}), loop)
                            
                            # Ao sair do loop de conversação
                            asyncio.run_coroutine_threadsafe(event_queue.put({"type": "sleep_mode"}), loop)
                            try: winsound.Beep(600, 200); winsound.Beep(400, 200)
                            except: pass
                            is_processing = False
        except Exception as e: 
            print(f"[ERR LISTENER] {e}")
            is_processing = False; time.sleep(2)

def load_config_from_disk():
    global mega_agent_instance
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            d = json.load(f); current_config.update(d)
            if d.get("apiKey"): mega_agent_instance = MegaAgent(current_config)

@app.on_event("startup")
def startup_event():
    load_config_from_disk()
    loop = asyncio.get_event_loop()
    threading.Thread(target=voice_listener_loop, args=(loop,), daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

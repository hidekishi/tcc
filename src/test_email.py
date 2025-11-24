#!/usr/bin/env python3
"""
Script de teste para verificar configuração de email
"""

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

def test_email_config(config_file='email_config.json'):
    """Testa o envio de email usando as configurações"""
    
    print("=" * 60)
    print("🧪 TESTE DE CONFIGURAÇÃO DE EMAIL")
    print("=" * 60)
    
    # Carregar configuração
    try:
        with open(config_file) as f:
            config = json.load(f)
        print(f"✅ Arquivo de configuração carregado: {config_file}")
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False
    
    # Validar campos obrigatórios
    required_fields = ['sender', 'password', 'recipients', 'smtp_server', 'smtp_port']
    for field in required_fields:
        if field not in config:
            print(f"❌ Campo obrigatório ausente: {field}")
            return False
    
    print(f"📧 Remetente: {config['sender']}")
    print(f"📬 Destinatários: {', '.join(config['recipients'])}")
    print(f"🌐 Servidor SMTP: {config['smtp_server']}:{config['smtp_port']}")
    print()
    
    # Criar mensagem de teste
    msg = MIMEMultipart()
    msg['From'] = config['sender']
    msg['To'] = ', '.join(config['recipients'])
    msg['Subject'] = f"[TESTE] Benchmark Runner - Teste de Email - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    body = f"""
🧪 TESTE DE CONFIGURAÇÃO DE EMAIL
==================================

Este é um email de teste do sistema de benchmark OpenMP.

✅ Configuração validada com sucesso!

Detalhes da configuração:
- Remetente: {config['sender']}
- Servidor SMTP: {config['smtp_server']}:{config['smtp_port']}
- Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Se você recebeu este email, a configuração está funcionando corretamente!

Próximo passo: Execute o benchmark completo com:
python benchmark_runner.py --full-test --email-notification --email-config email_config.json

---
Sistema de Benchmark OpenMP
"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Tentar enviar
    print("📤 Enviando email de teste...")
    try:
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.set_debuglevel(0)  # Desabilitar debug verbose
        server.starttls()
        print("🔐 Conectado ao servidor, autenticando...")
        
        server.login(config['sender'], config['password'])
        print("✅ Autenticação bem-sucedida!")
        
        text = msg.as_string()
        server.sendmail(config['sender'], config['recipients'], text)
        server.quit()
        
        print()
        print("=" * 60)
        print("✅ EMAIL ENVIADO COM SUCESSO!")
        print("=" * 60)
        print(f"📬 Verifique a caixa de entrada de: {', '.join(config['recipients'])}")
        print("💡 Pode levar alguns segundos para chegar")
        print()
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print()
        print("=" * 60)
        print("❌ ERRO DE AUTENTICAÇÃO")
        print("=" * 60)
        print(f"Detalhes: {e}")
        print()
        print("Possíveis causas:")
        print("1. Senha incorreta")
        print("2. Senha de app necessária (se tiver 2FA ativo)")
        print("3. Acesso de aplicativos menos seguros bloqueado")
        print()
        print("Para Microsoft/Outlook:")
        print("- Gere uma senha de app em: https://account.microsoft.com/security")
        print("- Ative 'Verificação em duas etapas' primeiro")
        print()
        return False
        
    except smtplib.SMTPException as e:
        print()
        print("=" * 60)
        print("❌ ERRO SMTP")
        print("=" * 60)
        print(f"Detalhes: {e}")
        print()
        print("Verifique:")
        print(f"- Servidor SMTP: {config['smtp_server']}")
        print(f"- Porta: {config['smtp_port']}")
        print("- Conexão de internet")
        print()
        return False
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ ERRO DESCONHECIDO")
        print("=" * 60)
        print(f"Detalhes: {e}")
        print()
        return False

if __name__ == '__main__':
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'email_config.json'
    success = test_email_config(config_file)
    
    sys.exit(0 if success else 1)

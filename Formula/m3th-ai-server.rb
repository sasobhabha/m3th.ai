class M3thAiServer < Formula
  desc "AI math practice app with Qwen LoRA (Flask server)"
  homepage "https://github.com/sasobhabha/m3th.ai"
  url "https://github.com/sasobhabha/m3th.ai.git", branch: "main"
  version "0.3.0"

  depends_on "uv"
  depends_on "python@3.12"

  def install
    libexec.install Dir["*"]
    
    (bin/"m3th-ai-server").write <<~EOS
      #!/bin/bash
      cd #{libexec}
      # Run via uv to handle all the heavy ML dependencies at runtime
      exec uv run --extra web python3 app.py "$@"
    EOS
  end
end

class M3th < Formula
  include Language::Python::Virtualenv

  desc "AMC 10-style problem generator and quiz — char-level GPT trained from scratch"
  homepage "https://github.com/sasobhabha/m3th.ai"
  url "https://github.com/sasobhabha/m3th.ai/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "57f012c4656ef8bbe83de0add2bb063b416740a73914356692c5477a0a0c9dd1"
  version "0.1.1"

  depends_on "python@3.12"
  depends_on macos: :sonoma
  depends_on arch: :arm64

  # Pretrained model weights (served as a GitHub release asset).
  resource "checkpoint" do
    url "https://github.com/sasobhabha/m3th.ai/releases/download/v0.1.1/best.pt"
    sha256 "4eadad04161fde5bd4c8324263ff64f9d518dabaf18c8e86bb70836f7b0cdae2"
  end

  # Pinned cp312 macOS arm64 wheels. Brew fetches and sha256-verifies each.
  # Installed in dependency order with --no-deps.
  resource "setuptools" do
    url "https://files.pythonhosted.org/packages/69/8a/b9dc7678803429e4a3bc9ba462fa3dd9066824d3c607490235c6a796be5a/setuptools-75.8.0-py3-none-any.whl"
    sha256 "e3982f444617239225d675215d51f6ba05f845d4eec313da4418fdbb56fb27e3"
  end

  resource "mpmath" do
    url "https://files.pythonhosted.org/packages/43/e3/7d92a15f894aa0c9c4b49b8ee9ac9850d6e63b03c9c32c0367a13ae62209/mpmath-1.3.0-py3-none-any.whl"
    sha256 "a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c"
  end

  resource "filelock" do
    url "https://files.pythonhosted.org/packages/36/d2/b70a31e13d04456d28493f31d2aa087e99eeb2767ef0293b2625727ccb8c/filelock-3.32.5-py3-none-any.whl"
    sha256 "142cd9fa77a872c5e78c62329a0d15278fadc686eb89e760017968961a4fd6b2"
  end

  resource "typing_extensions" do
    url "https://files.pythonhosted.org/packages/49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl"
    sha256 "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"
  end

  resource "fsspec" do
    url "https://files.pythonhosted.org/packages/fd/3c/6a2bf344106328fd04963664a60b9bb6496fc25df8e962fcdc1367285fb9/fsspec-2026.7.0-py3-none-any.whl"
    sha256 "b57ddbafedfaef7018c1ecab32aa200a9d7ca26b77965f64e48b70061249d279"
  end

  resource "networkx" do
    url "https://files.pythonhosted.org/packages/9e/c9/b2622292ea83fbb4ec318f5b9ab867d0a28ab43c5717bb85b0a5f6b3b0a4/networkx-3.6.1-py3-none-any.whl"
    sha256 "d47fbf302e7d9cbbb9e2555a0d267983d2aa476bac30e90dfbe5669bd57f3762"
  end

  resource "markupsafe" do
    url "https://files.pythonhosted.org/packages/9a/81/7e4e08678a1f98521201c3079f77db69fb552acd56067661f8c2f534a718/markupsafe-3.0.3-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "1872df69a4de6aead3491198eaf13810b565bdbeec3ae2dc8780f14458ec73ce"
  end

  resource "jinja2" do
    url "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl"
    sha256 "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67"
  end

  resource "sympy" do
    url "https://files.pythonhosted.org/packages/a2/09/77d55d46fd61b4a135c444fc97158ef34a095e5681d0a6c10b75bf356191/sympy-1.14.0-py3-none-any.whl"
    sha256 "e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/60/39/789131c1188c078dcb3a1692e72e1e050c68b88ffe72c9ccaac9bcd7a9cd/numpy-2.5.3-cp312-cp312-macosx_11_0_arm64.whl"
    sha256 "f59a878c33d6b88122d80d239bb3b845d58708750b0cb06a09aebb9b18ec696c"
  end

  resource "torch" do
    url "https://files.pythonhosted.org/packages/17/76/bb4770f56cf6d8971671dbcbb7493e5a6a15ad2825f4e359b02c27c38297/torch-2.14.0-cp312-cp312-macosx_14_0_arm64.whl"
    sha256 "c1f844f1c750e87df4b68bc3afbc0e2b0c7ef19d7b8f666e48bdcf6a0c4f0056"
  end

  WHEEL_ORDER = %w[
    setuptools mpmath filelock typing_extensions fsspec networkx
    markupsafe jinja2 sympy numpy torch
  ].freeze

  def pip_into(venv_python, requirements)
    # Drive the brewed interpreter's pip against the venv (the venv itself has
    # no pip, per Homebrew's Python 3.12+ convention).
    system Formula["python@3.12"].opt_bin/"python3.12", "-m", "pip",
           "--python=#{venv_python}", "install", "--no-cache-dir", *requirements
  end

  def install
    # Pretrained model weights, loaded via M3TH_CKPT by the bin wrapper.
    resource("checkpoint").stage do
      (libexec/"share").install "best.pt"
    end

    venv = virtualenv_create(libexec/"venv", "python3.12", system_site_packages: false)

    # Install each prebuilt wheel (deps pinned in WHEEL_ORDER, so --no-deps).
    WHEEL_ORDER.each do |name|
      resource(name).stage do
        wheel = Dir["*.whl"].first
        pip_into(venv.root/"bin/python", ["--no-deps", wheel])
      end
    end

    # Install the m3th package itself from the buildpath (deps pre-installed).
    pip_into(venv.root/"bin/python", ["--no-deps", buildpath])

    # Re-wrap the console script with M3TH_CKPT pointing at the bundled weights.
    (bin/"m3th").unlink if (bin/"m3th").exist?
    (bin/"m3th").write_env_script(libexec/"venv"/"bin"/"m3th", M3TH_CKPT: libexec/"share"/"best.pt")
  end

  def caveats
    <<~EOS
      A pretrained checkpoint is bundled automatically.

      Usage:
        m3th          generate AMC 10-style problems (interactive REPL)
        m3th quiz     quiz mode: answers are checked against the official key
    EOS
  end

  test do
    assert_match "quiz", shell_output("#{bin}/m3th --help")
  end
end

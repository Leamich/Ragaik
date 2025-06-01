<div class="flex flex-col min-h-screen bg-gray-50">
  <!-- Header -->
  <header class="bg-white shadow-md">
    <nav class="container mx-auto px-6 py-4 flex justify-between items-center">
    <a href="/" class="text-3xl font-bold text-violet-400">
        RAGaik
    </a>
      <div class="flex space-x-4">
        <a href="/chats"
            class="px-4 py-2 bg-gray-100 text-violet-400 rounded-md font-medium hover:bg-gray-200 transition-colors duration-200">
          Чат
        </a>
        <a href="/contacts"
           class="px-4 py-2 bg-gray-100 text-violet-400 rounded-md font-medium hover:bg-gray-200 transition-colors duration-200">
          Контакты
        </a>
      </div>
    </nav>
  </header>

  <!-- Hero Section -->
  <main class="flex-grow">
    <section class="bg-white">
      <div class="container mx-auto px-6 py-20 text-center">
        <h1 class="text-4xl sm:text-5xl md:text-6xl font-extrabold text-gray-800 leading-tight">
            Welcome to RAGaik!
        </h1>
        <p class="mt-6 text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto">
            Этот чат-бот – верный помощник в мире кучи учебников и конспектов
            
        </p>
        <div class="mt-8 flex justify-center space-x-4">
          <a href="/chats"
             class="px-6 py-3 bg-violet-400 text-white rounded-md font-medium hover:bg-violet-600 transition-colors duration-200">
            Начать
          </a>
          <a href="#features"
             class="px-6 py-3 bg-transparent border-2 border-violet-400 text-violet-400 rounded-md font-medium hover:bg-violet-50 transition-colors duration-200">
            Узнать больше
          </a>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section id="features" class="bg-gray-50">
      <div class="container mx-auto px-6 py-16">
        <h2 class="text-3xl font-bold text-gray-800 text-center">
            Преимущества нашего проекта
        </h2>
        <div class="mt-12 grid gap-8 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
          <!-- Карточка 1 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-start">
            <div class="flex items-center">
                <svg class="h-12 w-12 text-violet-400 mb-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" transform="scale(-1, 1)">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M21 21l-4.35-4.35" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                d="M10 18a8 8 0 100-16 8 8 0 000 16z" />
                </svg>
                <span class="text-violet-400 text-lg font-bold italic">E = mc²</span>
            </div>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
                Информация из проверенных источников
            </h3>
            <p class="text-gray-600">
                Чат-бот использует информацию только из одобренных источников, поэтому точно не ошибется!<a href="#footnote-1" class="text-gray-400"><sup>1</sup></a>
            </p>
          </div>
          <!-- Карточка 2 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-start">
            <svg class="h-12 w-12 text-violet-400 mb-4" fill="none" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M4 3h13a2 2 0 012 2v16l-4-2-4 2-4-2-4 2V5a2 2 0 012-2z" fill="currentColor" />
                <path d="M6 6h9v2H6V6zm0 4h9v2H6v-2zm0 4h9v2H6v-2z" fill="#fff" />
                <path d="M17 3v9l-2.5-1.5L12 12V3h5z" fill="currentColor" stroke="#fff" stroke-width="1" />
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
                Поддержка русского языка
            </h3>
            <p class="text-gray-600">
                Теперь для изучения твоего любимого предмета не нужно покупать русско-китайский словарь! Достаточно спросить RAGaik
            </p>
          </div>
          <!-- Карточка 3 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-start">
            <svg class="h-12 w-12 text-violet-400 mb-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" fill="currentColor">
                <path d="M64 4C31.4 4 4 31.4 4 64s27.4 60 60 60 60-27.4 60-60S96.6 4 64 4zm0 112c-28.7 0-52-23.3-52-52S35.3 12 64 12s52 23.3 52 52-23.3 52-52 52z"/>
                <text x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="36" font-family="serif" fill="currentColor">
                    LaTeX
                </text>
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
                Поддержка LaTeX
            </h3>
            <p class="text-gray-600">
                Пиши формулы и уравнения так как хочешь ты, RAGaik всегда ответит в виде красивой матформулы, а не непонятных символов! Где еще такое видано?
            </p>
        </div>
      </div>
    </section>

    <!-- Call-to-Action Section -->
    <section class="bg-violet-400">
      <div class="container mx-auto px-6 py-16 text-center text-white">
        <h2 class="text-3xl font-bold">
          Захотелось попробовать?
        </h2>
        <p class="mt-4 text-lg max-w-2xl mx-auto">
            Открывай свой список ДЗ и выбирай самое сложное. Наш чат-бот уже ждет твоих вопросов!
        </p>
        <a href="/chats"
           class="mt-8 inline-block px-8 py-4 bg-white text-violet-400 font-semibold rounded-md hover:bg-gray-100 transition-colors duration-200">
            Открыть чат
        </a>
      </div>
    </section>
  </main>

  <!-- Footer -->
<footer class="bg-white shadow-inner">
    <div class="container mx-auto px-6 py-4 text-left text-gray-600 text-sm">
        <a id="footnote-1"></a>
        <sup>1</sup> Данный чат-бот может содержать ошибки. Используйте его с осторожностью.
    </div>
    <div class="container mx-auto px-6 py-5 text-center text-gray-600 text-sm">
      © {new Date().getFullYear()} RAGaik. Не все права защищены.
    </div>
</footer>
</div>

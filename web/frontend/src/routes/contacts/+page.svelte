<!-- src/routes/contacts/+page.svelte -->
<script>
	// Если нужно динамически подставлять год в футере:
	let currentYear = new Date().getFullYear();
</script>

<div class="flex flex-col min-h-screen bg-gray-50">
  <!-- Header -->
  <header class="bg-white shadow-md">
    <nav class="container mx-auto px-6 py-4 flex justify-between items-center">
      <a href="/" class="text-3xl font-bold text-violet-400">
          RAGaik
      </a>
      <div class="flex space-x-4">
        <a
          href="/chats"
          class="px-4 py-2 bg-gray-100 text-violet-400 rounded-md font-medium hover:bg-gray-200 transition-colors duration-200"
        >
          Чат
        </a>
        <a
          href="/contacts"
          class="px-4 py-2 bg-violet-400 text-white rounded-md font-medium hover:bg-violet-500 transition-colors duration-200"
        >
          Контакты
        </a>
      </div>
    </nav>
  </header>

  <!-- Main Content -->
  <main class="flex-grow">
    <!-- QR Code Section -->
    <section class="bg-white">
      <div class="container mx-auto px-6 py-20 text-center">
        <h1 class="text-4xl sm:text-5xl md:text-6xl font-extrabold text-gray-800 leading-tight mb-8">
          Наш GitHub репозиторий
        </h1>
        <div class="inline-block bg-gray-100 p-6 rounded-lg shadow-lg">
          <a
            href="https://github.com/Leamich/Ragaik"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="/images/QR_GH.png"
              alt="QR-код на GitHub репозиторий"
              class="w-64 h-64 mx-auto"
            />
          </a>
          <p class="mt-4 text-lg text-gray-600">
            Наведите камеру и перейдите<br />
            <a
              href="https://github.com/your-org/your-repo"
              target="_blank"
              rel="noopener noreferrer"
              class="text-violet-400 hover:underline"
            >
              на страницу репозитория
            </a>
          </p>
        </div>
      </div>
    </section>

    <!-- Создатели Section -->
    <section class="bg-gray-50">
      <div class="container mx-auto px-6 py-16">
        <h2 class="text-3xl font-bold text-gray-800 text-center mb-12">
          Создатели проекта
        </h2>
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          <!-- Автор 1 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-center">
            <!-- GitHub Icon -->
            <svg
              class="h-12 w-12 text-gray-800 mb-4"
              role="img"
              viewBox="0 0 24 24"
              fill="currentColor"
              xmlns="http://www.w3.org/2000/svg"
            >
              <title>GitHub icon</title>
              <path
                d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 
                   3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 
                   0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61 
                   -.546-1.387-1.333-1.757-1.333-1.757-1.09-.744.084-.729.084-.729 
                   1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.807 1.305 3.492.998 
                   .108-.776.418-1.305.76-1.605-2.665-.3-5.467-1.332-5.467-5.93 
                   0-1.31.467-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 
                   0 0 1.005-.322 3.3 1.23a11.52 11.52 0 013.003-.404c1.02.005 
                   2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23 
                   .645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 
                   0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 
                   0 1.606-.015 2.896-.015 3.286 0 .315.21.694.825.576C20.565 
                   22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
              />
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
              <a
                href="https://github.com/Leamich"
                target="_blank"
                rel="noopener noreferrer"
                class="hover:text-violet-400"
              >
                Leamich
              </a>
            </h3>
            <p class="text-gray-600 text-sm">Mikhail Leontyev</p>
          </div>
          <!-- Автор 2 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-center">
            <svg
              class="h-12 w-12 text-gray-800 mb-4"
              role="img"
              viewBox="0 0 24 24"
              fill="currentColor"
              xmlns="http://www.w3.org/2000/svg"
            >
              <title>GitHub icon</title>
              <path
                d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 
                   3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 
                   0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61 
                   -.546-1.387-1.333-1.757-1.333-1.757-1.09-.744.084-.729.084-.729 
                   1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.807 1.305 3.492.998 
                   .108-.776.418-1.305.76-1.605-2.665-.3-5.467-1.332-5.467-5.93 
                   0-1.31.467-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 
                   0 0 1.005-.322 3.3 1.23a11.52 11.52 0 013.003-.404c1.02.005 
                   2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23 
                   .645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 
                   0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 
                   0 1.606-.015 2.896-.015 3.286 0 .315.21.694.825.576C20.565 
                   22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
              />
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
              <a
                href="https://github.com/le9endwp"
                target="_blank"
                rel="noopener noreferrer"
                class="hover:text-violet-400"
              >
                le9endwp
              </a>
            </h3>
            <p class="text-gray-600 text-sm">Sergey Shorokhov</p>
          </div>
          <!-- Автор 3 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-center">
            <svg
              class="h-12 w-12 text-gray-800 mb-4"
              role="img"
              viewBox="0 0 24 24"
              fill="currentColor"
              xmlns="http://www.w3.org/2000/svg"
            >
              <title>GitHub icon</title>
              <path
                d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 
                   3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 
                   0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61 
                   -.546-1.387-1.333-1.757-1.333-1.757-1.09-.744.084-.729.084-.729 
                   1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.807 1.305 3.492.998 
                   .108-.776.418-1.305.76-1.605-2.665-.3-5.467-1.332-5.467-5.93 
                   0-1.31.467-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 
                   0 0 1.005-.322 3.3 1.23a11.52 11.52 0 013.003-.404c1.02.005 
                   2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23 
                   .645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 
                   0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 
                   0 1.606-.015 2.896-.015 3.286 0 .315.21.694.825.576C20.565 
                   22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
              />
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
              <a
                href="https://github.com/TheKostins"
                target="_blank"
                rel="noopener noreferrer"
                class="hover:text-violet-400"
              >
                TheKostins
              </a>
            </h3>
            <p class="text-gray-600 text-sm">Konstantin Anisimov</p>
          </div>
          <!-- Автор 4 -->
          <div class="bg-white rounded-lg shadow-md p-6 flex flex-col items-center">
            <svg
              class="h-12 w-12 text-gray-800 mb-4"
              role="img"
              viewBox="0 0 24 24"
              fill="currentColor"
              xmlns="http://www.w3.org/2000/svg"
            >
              <title>GitHub icon</title>
              <path
                d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 
                   3.438 9.8 8.205 11.387.6.113.82-.258.82-.577 
                   0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61 
                   -.546-1.387-1.333-1.757-1.333-1.757-1.09-.744.084-.729.084-.729 
                   1.205.084 1.84 1.236 1.84 1.236 1.07 1.835 2.807 1.305 3.492.998 
                   .108-.776.418-1.305.76-1.605-2.665-.3-5.467-1.332-5.467-5.93 
                   0-1.31.467-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 
                   0 0 1.005-.322 3.3 1.23a11.52 11.52 0 013.003-.404c1.02.005 
                   2.045.138 3.003.404 2.28-1.552 3.285-1.23 3.285-1.23 
                   .645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 
                   0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 
                   0 1.606-.015 2.896-.015 3.286 0 .315.21.694.825.576C20.565 
                   22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"
              />
            </svg>
            <h3 class="text-xl font-semibold text-gray-800 mb-2">
              <a
                href="https://github.com/ubetu"
                target="_blank"
                rel="noopener noreferrer"
                class="hover:text-violet-400"
              >
                ubetu
              </a>
            </h3>
            <p class="text-gray-600 text-sm">Pavel Seroglazov</p>
          </div>
        </div>
      </div>
    </section>
  </main>

  <!-- Footer -->
  <footer class="bg-white shadow-inner">
    <div class="container mx-auto px-6 py-4 text-left text-gray-600 text-sm">
      <sup>1</sup> Данный чат-бот может содержать ошибки. Используйте его с осторожностью.
    </div>
    <div class="container mx-auto px-6 py-5 text-center text-gray-600 text-sm">
      © {currentYear} RAGaik. Не все права защищены.
    </div>
  </footer>
</div>

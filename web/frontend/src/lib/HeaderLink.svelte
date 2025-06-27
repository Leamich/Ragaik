<script lang="ts">
	import {page} from '$app/stores';

	export let href: string;
	export let text: string;

    let inactiveClasses = 'bg-gray-100 text-violet-400 hover:bg-gray-200';
    let activeClasses = 'bg-violet-400 text-white hover:bg-violet-500';
</script>

<a
	{href}
	class={`px-4 py-2 rounded-md font-medium transition-colors duration-200 ${
		$page.url.pathname === href ? activeClasses : inactiveClasses
	}`}
>
	{text}
</a>

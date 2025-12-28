document.addEventListener('DOMContentLoaded', function(){
  // 导航菜单切换
  const btn = document.getElementById('nav-toggle');
  const menu = document.getElementById('nav-menu');

  if(btn && menu) {
    btn.addEventListener('click', function(){
      const isOpen = menu.classList.toggle('open');
      btn.setAttribute('aria-expanded', isOpen);
      btn.innerHTML = isOpen ? '✕' : '☰';
    });

    // 小屏幕点击外部关闭菜单
    document.addEventListener('click', function(e){
      if(window.innerWidth > 800) return;
      if(!menu.classList.contains('open')) return;
      if(!menu.contains(e.target) && e.target !== btn){
        menu.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
        btn.innerHTML = '☰';
      }
    });
  }

  // 为当前页面导航项添加active类
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.site-nav a');

  navLinks.forEach(link => {
    if(link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });

  // 添加滚动时的导航栏效果
  const header = document.querySelector('.site-header');
  let lastScrollTop = 0;

  window.addEventListener('scroll', function() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    if(scrollTop > 50) {
      header.style.padding = '12px 0';
      header.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)';
    } else {
      header.style.padding = '18px 0';
      header.style.boxShadow = 'var(--shadow-light)';
    }

    lastScrollTop = scrollTop;
  });
});
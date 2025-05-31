// GSAP Animations
gsap.registerPlugin(ScrollTrigger);

// Initialize animations when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Hero animations
    const heroTimeline = gsap.timeline({ delay: 1.5 });
    
    heroTimeline
        .from('.hero-title', {
            opacity: 0,
            y: 50,
            duration: 1,
            ease: 'power3.out'
        })
        .from('.hero-subtitle', {
            opacity: 0,
            y: 30,
            duration: 0.8,
            ease: 'power3.out'
        }, '-=0.3')
        .from('.hero-description', {
            opacity: 0,
            y: 30,
            duration: 0.8,
            ease: 'power3.out'
        }, '-=0.3')
        .from('.hero-stats .stat', {
            opacity: 0,
            y: 30,
            duration: 0.6,
            stagger: 0.2,
            ease: 'power3.out'
        }, '-=0.3')
        .from('.hero-buttons .btn', {
            opacity: 0,
            y: 30,
            duration: 0.6,
            stagger: 0.1,
            ease: 'power3.out'
        }, '-=0.3')
        .from('.floating-card', {
            opacity: 0,
            scale: 0,
            duration: 0.8,
            stagger: 0.2,
            ease: 'back.out(1.7)'
        }, '-=0.5');

    // Section animations
    gsap.utils.toArray('.section').forEach((section, index) => {
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: section,
                start: 'top 80%',
                end: 'bottom 20%',
                toggleActions: 'play none none reverse'
            }
        });

        // Animate section headers
        const header = section.querySelector('.section-header');
        if (header) {
            tl.from(header.querySelector('.section-title'), {
                opacity: 0,
                y: 50,
                duration: 0.8,
                ease: 'power3.out'
            })
            .from(header.querySelector('.section-subtitle'), {
                opacity: 0,
                y: 30,
                duration: 0.6,
                ease: 'power3.out'
            }, '-=0.3');
        }
    });

    // Timeline animations
    gsap.utils.toArray('.timeline-item').forEach((item, index) => {
        gsap.from(item, {
            opacity: 0,
            x: index % 2 === 0 ? -100 : 100,
            duration: 0.8,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: item,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Project cards animations
    gsap.utils.toArray('.project-card').forEach((card, index) => {
        gsap.from(card, {
            opacity: 0,
            y: 100,
            duration: 0.8,
            delay: index * 0.2,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: card,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });

        // Hover animations
        card.addEventListener('mouseenter', () => {
            gsap.to(card, {
                y: -10,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                y: 0,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });

    // Skills animation
    gsap.utils.toArray('.skill-item').forEach((skill, index) => {
        gsap.from(skill, {
            opacity: 0,
            scale: 0,
            duration: 0.5,
            delay: index * 0.05,
            ease: 'back.out(1.7)',
            scrollTrigger: {
                trigger: skill.closest('.skills-container'),
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Research card animation
    const researchCard = document.querySelector('.research-card');
    if (researchCard) {
        gsap.from(researchCard, {
            opacity: 0,
            scale: 0.8,
            duration: 1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: researchCard,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    }

    // Metrics animation
    gsap.utils.toArray('.metric').forEach((metric, index) => {
        gsap.from(metric, {
            opacity: 0,
            scale: 0,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'back.out(1.7)',
            scrollTrigger: {
                trigger: metric.closest('.impact-metrics, .publication-metrics'),
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Contact form animations
    gsap.utils.toArray('.form-group').forEach((group, index) => {
        gsap.from(group, {
            opacity: 0,
            x: -50,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.contact-form',
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Certification items animation
    gsap.utils.toArray('.certification-item').forEach((item, index) => {
        gsap.from(item, {
            opacity: 0,
            y: 50,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: item.closest('.certifications-grid'),
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Floating animation for cards
    gsap.utils.toArray('.floating-card').forEach((card, index) => {
        gsap.to(card, {
            y: '-20px',
            duration: 2 + index * 0.5,
            ease: 'power1.inOut',
            yoyo: true,
            repeat: -1
        });
    });

    // Parallax effect for background elements
    gsap.to('#particles-js', {
        yPercent: -50,
        ease: 'none',
        scrollTrigger: {
            trigger: 'body',
            start: 'top top',
            end: 'bottom top',
            scrub: true
        }
    });

    // Navigation animation on scroll
    ScrollTrigger.create({
        start: 'top -80',
        end: 99999,
        toggleClass: {
            className: 'nav-scrolled',
            targets: '.navbar'
        }
    });

    // Button hover animations
    gsap.utils.toArray('.btn').forEach(btn => {
        btn.addEventListener('mouseenter', () => {
            gsap.to(btn, {
                scale: 1.05,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        btn.addEventListener('mouseleave', () => {
            gsap.to(btn, {
                scale: 1,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });

    // Text reveal animation
    gsap.utils.toArray('.gradient-text').forEach(text => {
        gsap.from(text, {
            backgroundPosition: '200% center',
            duration: 2,
            ease: 'power2.out',
            scrollTrigger: {
                trigger: text,
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Loading animation for tech tags
    gsap.utils.toArray('.tech-tag').forEach((tag, index) => {
        gsap.from(tag, {
            opacity: 0,
            scale: 0,
            duration: 0.4,
            delay: index * 0.05,
            ease: 'back.out(1.7)',
            scrollTrigger: {
                trigger: tag.closest('.experience-tech, .project-tech'),
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });
    });

    // Social links animation
    gsap.utils.toArray('.social-link').forEach((link, index) => {
        gsap.from(link, {
            opacity: 0,
            y: 30,
            duration: 0.6,
            delay: index * 0.1,
            ease: 'back.out(1.7)',
            scrollTrigger: {
                trigger: '.social-links',
                start: 'top 80%',
                toggleActions: 'play none none reverse'
            }
        });

        // Hover effect
        link.addEventListener('mouseenter', () => {
            gsap.to(link, {
                scale: 1.1,
                y: -5,
                duration: 0.3,
                ease: 'power2.out'
            });
        });

        link.addEventListener('mouseleave', () => {
            gsap.to(link, {
                scale: 1,
                y: 0,
                duration: 0.3,
                ease: 'power2.out'
            });
        });
    });
});

// Custom cursor animations
document.addEventListener('mousemove', (e) => {
    gsap.to('.cursor', {
        x: e.clientX,
        y: e.clientY,
        duration: 0.1,
        ease: 'power2.out'
    });
    
    gsap.to('.cursor-follower', {
        x: e.clientX,
        y: e.clientY,
        duration: 0.3,
        ease: 'power2.out'
    });
});

// Cursor hover effects
document.querySelectorAll('a, button, .btn').forEach(el => {
    el.addEventListener('mouseenter', () => {
        gsap.to('.cursor', {
            scale: 2,
            duration: 0.3,
            ease: 'power2.out'
        });
        gsap.to('.cursor-follower', {
            scale: 1.5,
            duration: 0.3,
            ease: 'power2.out'
        });
    });
    
    el.addEventListener('mouseleave', () => {
        gsap.to('.cursor', {
            scale: 1,
            duration: 0.3,
            ease: 'power2.out'
        });
        gsap.to('.cursor-follower', {
            scale: 1,
            duration: 0.3,
            ease: 'power2.out'
        });
    });
});

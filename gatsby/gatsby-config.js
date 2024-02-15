module.exports = {
  pathPrefix: `/catalogs/default`,
  siteMetadata: {
    title: 'album catalog',
    subtitle: 'sharing favourite solutions across tools and domains',
    catalog_url: 'https://gitlab.com/album-app/catalogs/templates/catalog-gatsby',
    menuLinks:[
      {
         name:'Catalog',
         link:'/catalog'
      },
      {
         name:'About',
         link:'/about'
      },
    ]
  },
  plugins: [{ resolve: `gatsby-theme-album`, options: {} }],
}
